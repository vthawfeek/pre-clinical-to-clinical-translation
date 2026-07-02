import torch

from pctrans.models.dual_tower import DualTowerModel
from pctrans.models.encoders import CCLEEncoder, MLPBlock, TCGAEncoder


def test_encoder_output_shape(tiny_model):
    x = torch.randn(8, 2000)
    z = tiny_model.encode_ccle(x)
    assert z.shape == (8, 64)


def test_l2_norm_unit_sphere(tiny_model):
    x = torch.randn(8, 2000)
    z = tiny_model.encode_ccle(x)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_forward_returns_two_normalised_towers(tiny_model):
    x_ccle = torch.randn(8, 2000)
    x_tcga = torch.randn(12, 2000)
    z_ccle, z_tcga = tiny_model(x_ccle, x_tcga)
    assert z_ccle.shape == (8, 64)
    assert z_tcga.shape == (12, 64)
    assert torch.allclose(z_ccle.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(z_tcga.norm(dim=-1), torch.ones(12), atol=1e-5)


def test_mlp_block_structure():
    block = MLPBlock(10, 4, dropout=0.3, use_bn=True)
    x = torch.randn(6, 10)
    assert block(x).shape == (6, 4)
    # Linear + BatchNorm1d + ReLU + Dropout
    assert len(block.block) == 4


def test_projection_head_has_no_batchnorm(tiny_model):
    # Final layer of each tower is a bare Linear (projection head).
    assert isinstance(tiny_model.ccle_encoder.projection, torch.nn.Linear)
    assert isinstance(tiny_model.tcga_encoder.projection, torch.nn.Linear)


def test_towers_have_separate_weights():
    ccle = CCLEEncoder(embed_dim=64)
    tcga = TCGAEncoder(embed_dim=64)
    # Same architecture, but independent parameter tensors.
    ccle_first = ccle.blocks[0].block[0].weight
    tcga_first = tcga.blocks[0].block[0].weight
    assert ccle_first.data_ptr() != tcga_first.data_ptr()


def test_gradient_flows_to_both_towers(tiny_model):
    x_ccle = torch.randn(6, 2000)
    x_tcga = torch.randn(6, 2000)
    z_ccle, z_tcga = tiny_model.train()(x_ccle, x_tcga)
    (z_ccle.sum() + z_tcga.sum()).backward()
    ccle_grad = tiny_model.ccle_encoder.projection.weight.grad
    tcga_grad = tiny_model.tcga_encoder.projection.weight.grad
    assert ccle_grad is not None and ccle_grad.abs().sum() > 0
    assert tcga_grad is not None and tcga_grad.abs().sum() > 0


def test_total_param_count_around_5_5m():
    model = DualTowerModel(CCLEEncoder(), TCGAEncoder())
    total = sum(p.numel() for p in model.parameters())
    # ~2.75M per encoder, ~5.5M total for the [2000,1024,512,256,128,64] template.
    assert 5_000_000 < total < 6_000_000
