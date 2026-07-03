"""Contrastive training loop for the dual-tower model.

`ContrastiveTrainer` connects the pieces built on Days 5-6: the stratified batch
sampler feeds lineage-balanced CCLE/TCGA mini-batches through `DualTowerModel`,
`SupConInfoNCELoss` scores them, and `KNNValidationCallback` measures cross-domain
retrieval each epoch. The best checkpoint (by validation kNN@5) is saved and
early stopping fires after `early_stop_patience` epochs without improvement.

Optimisation follows the plan: Adam, gradient-norm clipping to `grad_clip_norm`,
and a per-epoch cosine learning-rate schedule with a linear warmup of
`warmup_epochs`. MLflow logging is enabled only when a `mlflow_run_name` is
supplied, so unit tests run without writing an experiment.
"""

import contextlib
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pctrans.training.callbacks import KNNValidationCallback


class ContrastiveTrainer:
    def __init__(
        self,
        model,
        loss_fn,
        train_sampler,
        val_ccle,
        val_tcga,
        config,
        mlflow_run_name=None,
        idx_to_lineage=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_sampler = train_sampler
        self.val_ccle = val_ccle
        self.val_tcga = val_tcga
        self.config = dict(config or {})
        self.mlflow_run_name = mlflow_run_name

        self.lr = float(self.config.get("lr", 3.0e-4))
        self.warmup_epochs = int(self.config.get("warmup_epochs", 5))
        self.grad_clip_norm = float(self.config.get("grad_clip_norm", 1.0))
        self.knn_k = int(self.config.get("knn_k", 5))
        self.patience = int(self.config.get("early_stop_patience", 5))
        self.checkpoint_path = self.config.get("checkpoint_path", "models/best_model.pt")
        eval_batch_size = int(self.config.get("eval_batch_size", 256))

        self.knn_callback = KNNValidationCallback(k=self.knn_k, idx_to_lineage=idx_to_lineage)
        self._val_ccle_loader = DataLoader(val_ccle, batch_size=eval_batch_size, shuffle=False)
        self._val_tcga_loader = DataLoader(val_tcga, batch_size=eval_batch_size, shuffle=False)

    # -- learning-rate schedule -------------------------------------------------

    def _lr_lambda(self, n_epochs):
        def scale(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / max(1, self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / max(1, n_epochs - self.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return scale

    # -- MLflow (no-op unless a run name is given) ------------------------------

    @contextlib.contextmanager
    def _mlflow_run(self):
        if self.mlflow_run_name is None:
            yield None
            return
        import mlflow

        experiment = self.config.get("mlflow_experiment")
        if experiment:
            mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=self.mlflow_run_name) as run:
            yield run

    def _log_params(self, params):
        if self.mlflow_run_name is None:
            return
        import mlflow

        mlflow.log_params(params)

    def _log_metrics(self, metrics, step):
        if self.mlflow_run_name is None:
            return
        import mlflow

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    # -- one training epoch -----------------------------------------------------

    def _batch_tensors(self, dataset, indices):
        features = dataset.features[indices]
        labels = torch.as_tensor(dataset.labels[indices])
        return features, labels

    @staticmethod
    def _grad_norm(params):
        """L2 norm of the concatenated gradients of ``params`` (0.0 if none set)."""
        sq_sum = 0.0
        for p in params:
            if p.grad is not None:
                sq_sum += float(p.grad.detach().norm(2).item()) ** 2
        return sq_sum**0.5

    def _train_one_epoch(self, optimizer):
        self.model.train()
        ccle_ds = self.train_sampler.ccle_dataset
        tcga_ds = self.train_sampler.tcga_dataset
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        ccle_params = list(self.model.ccle_encoder.parameters())
        tcga_params = list(self.model.tcga_encoder.parameters())

        total_loss, n_batches = 0.0, 0
        ccle_grad_sum, tcga_grad_sum = 0.0, 0.0
        for batch in self.train_sampler:
            x_ccle, y_ccle = self._batch_tensors(ccle_ds, batch["ccle_indices"])
            x_tcga, y_tcga = self._batch_tensors(tcga_ds, batch["tcga_indices"])

            optimizer.zero_grad()
            z_ccle, z_tcga = self.model(x_ccle, x_tcga)
            loss = self.loss_fn(z_ccle, z_tcga, y_ccle, y_tcga)
            loss.backward()
            # Per-encoder gradient norm, measured pre-clip, for the Day 9
            # tower-asymmetry diagnostic (Panel 4).
            ccle_grad_sum += self._grad_norm(ccle_params)
            tcga_grad_sum += self._grad_norm(tcga_params)
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        denom = max(1, n_batches)
        return {
            "train_loss": total_loss / denom,
            "grad_norm_ccle": ccle_grad_sum / denom,
            "grad_norm_tcga": tcga_grad_sum / denom,
        }

    # -- validation loss (single pooled SupCon batch) --------------------------

    def _validation_loss(self):
        self.model.eval()
        with torch.no_grad():
            z_ccle = self.model.encode_ccle(self.val_ccle.features)
            z_tcga = self.model.encode_tcga(self.val_tcga.features)
            loss = self.loss_fn(
                z_ccle,
                z_tcga,
                torch.as_tensor(self.val_ccle.labels),
                torch.as_tensor(self.val_tcga.labels),
            )
        return float(loss.item())

    # -- checkpointing ----------------------------------------------------------

    def _save_checkpoint(self, epoch, val_knn):
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "loss_state_dict": self.loss_fn.state_dict(),
                "epoch": epoch,
                "val_knn_accuracy": val_knn,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "loss_state_dict" in checkpoint:
            self.loss_fn.load_state_dict(checkpoint["loss_state_dict"])
        return checkpoint

    # -- main loop --------------------------------------------------------------

    def train(self, n_epochs=None, on_epoch_end=None):
        """Fit the model.

        ``on_epoch_end``, if given, is called as ``on_epoch_end(epoch, model,
        record)`` after each epoch's validation. The model is in eval mode at
        that point; the hook is used by the Day 9 analysis notebook to snapshot
        validation embeddings without duplicating the training loop.
        """
        n_epochs = int(n_epochs or self.config.get("n_epochs", 30))
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda(n_epochs))

        history = []
        best_knn, best_epoch, epochs_since_improve = -1.0, -1, 0

        with self._mlflow_run():
            self._log_params(
                {
                    "lr": self.lr,
                    "n_epochs": n_epochs,
                    "warmup_epochs": self.warmup_epochs,
                    "grad_clip_norm": self.grad_clip_norm,
                    "knn_k": self.knn_k,
                    "batch_size": self.config.get("batch_size", "NA"),
                }
            )

            for epoch in range(n_epochs):
                epoch_stats = self._train_one_epoch(optimizer)
                train_loss = epoch_stats["train_loss"]
                scheduler.step()

                val_metrics = self.knn_callback(
                    self.model, self._val_ccle_loader, self._val_tcga_loader
                )
                val_knn = val_metrics["val_knn_accuracy"]
                val_loss = self._validation_loss()
                tau = float(self.loss_fn.tau.item())
                lr = optimizer.param_groups[0]["lr"]

                self._log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_knn_accuracy": val_knn,
                        "temperature": tau,
                        "lr": lr,
                        "grad_norm_ccle": epoch_stats["grad_norm_ccle"],
                        "grad_norm_tcga": epoch_stats["grad_norm_tcga"],
                    },
                    step=epoch,
                )

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_knn_accuracy": val_knn,
                    "temperature": tau,
                    "lr": lr,
                    "grad_norm_ccle": epoch_stats["grad_norm_ccle"],
                    "grad_norm_tcga": epoch_stats["grad_norm_tcga"],
                    "per_lineage": val_metrics["per_lineage"],
                }
                history.append(record)

                if on_epoch_end is not None:
                    on_epoch_end(epoch, self.model, record)

                if val_knn > best_knn:
                    best_knn, best_epoch, epochs_since_improve = val_knn, epoch, 0
                    self._save_checkpoint(epoch, val_knn)
                else:
                    epochs_since_improve += 1
                    if self.patience and epochs_since_improve >= self.patience:
                        break

        result = dict(history[-1]) if history else {}
        result["best_val_knn_accuracy"] = best_knn
        result["best_epoch"] = best_epoch
        result["history"] = history
        return result
