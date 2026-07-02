# Day 14: GitHub Release & Public Launch (Phase 1 wrap) + Phase 2 harness

**Date:** 2026-07-02
**Commit:** `day 14: v0.1.0 release, Colab quickstart, app artefacts tracked, Phase 2 harness wired`

## What Was Built

- **`requirements.txt`** — cross-platform runtime pins mirroring `pyproject.toml`'s top-level
  dependencies (torch, pandas, numpy, pyarrow, scikit-learn, umap-learn, numba/llvmlite, streamlit,
  plotly, matplotlib, mlflow, typer, pyyaml, requests, tqdm). Deliberately **not** a raw
  `uv pip freeze`: this venv is on Windows and a bare freeze would embed Windows-only wheels
  (pywin32 etc.) that fail to install on Streamlit Cloud's Linux runtime. Top-level pins let pip
  resolve the transitive tree per target platform (Linux/macOS/Windows), while `uv.lock` still
  provides the exact locked dev environment via `uv sync`.
- **`notebooks/colab_quickstart.ipynb`** — single-notebook, end-to-end run for a free Colab T4:
  install from GitHub → `pctrans-download` (CCLE + TCGA) → `pctrans-preprocess --split` →
  `pctrans-train` → `pctrans-evaluate` → `pctrans-visualize` with an inline UMAP render. 14 cells,
  validated as well-formed nbformat 4 JSON.
- **App artefacts now git-tracked** for Streamlit Cloud deployment: `.gitignore` changed from
  `data/processed/` to `data/processed/*` plus negations for the three files the app actually loads —
  `embeddings_test.npz` (122 KB), `ccle_embeddings.npz` (93 KB), `app_meta.json` (43 KB). The large
  parquets, scalers, splits, and `models/*.pt` stay ignored (verified via `git check-ignore`).
- **Final code review:** grep for absolute/hardcoded paths across `pctrans/`, `app/`, `tests/` — none
  found. The Streamlit app uses relative `Path("data/processed")`; no `C:\...`, `/home/...`, or
  `/tmp/...` literals in source.
- **`v0.1.0` git tag + GitHub release** marking end of Phase 1, with honest, scoped release notes (see
  below), and repository topics added.
- **Phase 2 harness wired** (the "arrangements" for `PLAN-phase2.md`):
  - `.claude/commands/day.md` now routes **Days 1–14 → `PLAN.md`, Days 15–25 → `PLAN-phase2.md`**,
    adds Day 25 as a blog-milestone and Day 24 as the Gate 2 day, and documents the
    keep-Phase-1-reproducible rule (new artefacts under `*_15` / `_trainhvg` prefixes).
  - `CLAUDE.md` references both plan files, widens the `/day` range to 1–25, and adds a **Phase 2
    status tracker** (Days 15–25, all PENDING).

## What Was Learned

- A Windows `uv pip freeze` is the wrong artefact for a Linux deployment target — the platform-marker
  distinction only surfaces when you actually think about where `requirements.txt` gets consumed
  (Streamlit Cloud, Colab), not where it was generated.
- Git negation cannot re-include a file whose *parent directory* is excluded; the directory has to be
  globbed (`data/processed/*`) rather than excluded (`data/processed/`) for `!file` rules to take
  effect. Confirmed with `git check-ignore -q`.
- The Streamlit app transitively imports `pctrans.data.dataset`, which imports `torch` — so torch is a
  genuine runtime dependency of the deployed app even though the app never loads a model checkpoint.
- The repo was already **public**, so no irreversible visibility flip was needed today.

## Key Decisions

- **Did not harden the README's bare "100%" claim today.** That is explicitly a Phase 2 **Day 24**
  task (replace with "100% (95% CI 90.8–100, n=38); stable across 10 seeds; beats
  Harmony/ComBat/Scanorama; survives label-shuffle and purity adjustment"), and the CI numbers do not
  exist until the Phase 2 stats work runs. Pre-empting it would mean inventing numbers.
- **Held the outward-facing launch actions that are irreversible, not-yet-earned, or not automatable,**
  and surfaced them as a manual checklist instead of firing them autonomously (see *Manual / deferred*).
  Given Phase 2 is specifically designed to stress-test and re-scope the Phase-1 "100%", broadcasting
  that number on social channels *before* the validation would be exactly the over-claim the next phase
  guards against. Recommendation: publish the social/blog launch after Gate 2 (Day 24), using the
  hardened claims and the Blog Post 3 credibility arc.
- **Tracked only the three app-required artefacts, not the model or parquets.** Keeps the repo lean;
  a fresh clone regenerates the model via the Colab quickstart / `pctrans-train`. Phase 2 runs locally
  where those files already exist.

## Verification

```
$ uv run ruff check pctrans/ tests/
All checks passed!

$ uv run pytest tests/ -q -m "not slow and not integration"
........................................................................ [ 97%]
..                                                                       [100%]
74 passed, 5 deselected in 13.48s

$ git check-ignore -q data/processed/embeddings_test.npz  ; echo trackable
trackable            # (also ccle_embeddings.npz, app_meta.json)
$ git check-ignore -q data/processed/ccle_2k.parquet      ; echo IGNORED
IGNORED              # large artefacts stay ignored

$ python -c "import json; nb=json.load(open('notebooks/colab_quickstart.ipynb',encoding='utf-8')); print(len(nb['cells']),'cells, nbformat',nb['nbformat'])"
14 cells, nbformat 4
```

**GitHub release v0.1.0 notes (Phase 1):**

> Phase 1 complete — dual-tower SupCon-InfoNCE alignment of CCLE cell lines to TCGA patients across
> LUAD/BRCA/SKCM. Held-out test set: kNN@5 = 100%, kNN@1 = 97.4%, silhouette +0.57, TFS 0.89
> (3-lineage task, n=38 test cell lines; baselines: random 33.3%, PCA+kNN 65.8%). Interactive demo:
> https://pctrans.streamlit.app · Colab quickstart in `notebooks/colab_quickstart.ipynb`.
> Note: these numbers are internally valid but not yet stress-tested — Phase 2 (`PLAN-phase2.md`) adds
> confidence intervals, real batch-correction baselines, a 15-lineage task, a purity confounder
> analysis, a label-shuffle negative control, and a vemurafenib/BRAF case study.

## Numbers (if applicable)

- Tests: **74 passed**, 5 deselected (slow/integration), ruff clean.
- App artefacts tracked: 3 files, ~258 KB total. Model (`best_model.pt`, 22 MB) and processed
  parquets (~17 MB) remain untracked.
- Colab notebook: 14 cells.
- Phase 2 tracker seeded: 11 days (15–25) marked PENDING.

## Manual / deferred (require a human or a web console)

1. **Streamlit Community Cloud connect** — one-time web action at share.streamlit.io: point it at this
   repo, main file `app/streamlit_app.py`, Python 3.11, `requirements.txt`. The app's data artefacts
   are now git-tracked, so the deploy has everything it needs. (App already advertised at
   https://pctrans.streamlit.app.)
2. **Social publishing** — LinkedIn Post 1/2 and the X thread drafts already live in `reports/`
   (`linkedin-01.txt`, `linkedin-02.txt`, `x-thread-02.txt`). Recommend posting **after** Phase 2
   Gate 2 so the headline numbers carry their confidence intervals and the "I tried to break it"
   Blog Post 3 arc.

## Next Up (Phase 2 — `PLAN-phase2.md`)

- **Day 15:** `pctrans/evaluation/stats.py` — Wilson + bootstrap CIs, `pctrans-multiseed` runner over
  10 seeds; attach an interval to every headline metric.
- **Day 16:** train-only HVG selection (`--hvg-on train`), leakage-delta vs. the Phase-1 all-sample
  gene list.
- **Day 17:** real Harmony/ComBat/Scanorama baselines on identical data + supervised cross-domain
  ceiling (optional `baselines` extra: harmonypy, scanorama, inmoose).
- **Day 18:** config-driven `LINEAGE_TO_IDX`, `configs/data_15.yaml`, 15-lineage training run
  (`best_model_15.pt`).
- Trigger with `/day 15` — it now reads `PLAN-phase2.md` automatically.
