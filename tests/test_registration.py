import json
from pathlib import Path

import numpy as np
import pytest
import torch

from axis.registration import (
    NCC,
    ImageTransitionNet,
    MaskedNCC,
    RegistrationLoss,
    TransitionFieldNet,
    ensure_flow_2hw,
    find_inverse_warp_coords,
    load_model,
    save_checkpoint,
    smoothness_loss,
    transport_points,
    upsample_flow,
    warp_image,
)

SCHEMA = json.loads((Path(__file__).parent / "data" / "checkpoint_schema.json").read_text())


@pytest.mark.parametrize(
    "key, model",
    [
        (
            "multimodal_transformer_rgb",
            lambda: TransitionFieldNet(in_channels=3, in_gene_channels=10, fusion_module="transformer"),
        ),
        ("multimodal_linear_gray", lambda: TransitionFieldNet(in_channels=1, in_gene_channels=10, fusion_module="linear")),
        ("image_rgb", lambda: ImageTransitionNet(in_channels=3)),
    ],
)
def test_checkpoint_schema_unchanged(key, model):
    """Parameter names/shapes must match the original networks so released checkpoints load."""
    state = {k: list(v.shape) for k, v in model().state_dict().items()}
    assert state == SCHEMA[key]


def _inputs(size=64, channels=3, comps=4):
    torch.manual_seed(0)
    image = lambda: torch.randn(1, channels, 1, size, size)  # noqa: E731
    gene = lambda: torch.rand(1, comps + 1, 1, size // 8, size // 8)  # noqa: E731
    return image(), gene(), image(), gene()


def test_forward_shapes_and_loss():
    model = TransitionFieldNet(in_channels=3, in_gene_channels=4)
    fi, fg, mi, mg = _inputs()
    out = model(fi, fg, mi, mg)
    warped_image, warped_gene, flow, flow_gene, affined_image, affined_gene, affine = out
    assert warped_image.shape == fi.shape and warped_gene.shape == mg.shape
    assert flow.shape == (1, 3, 1, 64, 64) and flow_gene.shape == (1, 3, 1, 8, 8)
    terms = RegistrationLoss(morph_win=5, mol_win=3, n_mol_channels=4)(fi, fg, out)
    assert all(torch.isfinite(v) for v in terms.values())
    terms["loss"].backward()
    assert model.predict_field(fi, fg, mi, mg).shape == (2, 64, 64)


def test_canvas_must_be_divisible_by_16():
    model = ImageTransitionNet(in_channels=1)
    x = torch.randn(1, 1, 1, 40, 40)
    with pytest.raises(ValueError):
        model(x, x)


def test_save_and_load_checkpoint(tmp_path):
    model = TransitionFieldNet(in_channels=1, in_gene_channels=3, fusion_module="linear")
    save_checkpoint(model, tmp_path / "m.pth")
    loaded = load_model(tmp_path / "m.pth", device="cpu")
    for (k, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k
    # plain state_dict (original format) with explicit architecture
    torch.save(model.state_dict(), tmp_path / "plain.pth")
    loaded = load_model(tmp_path / "plain.pth", device="cpu", in_channels=1, in_gene_channels=3, fusion_module="linear")
    assert isinstance(loaded, TransitionFieldNet)


def test_losses_identity():
    x = torch.rand(1, 3, 1, 32, 32)
    assert NCC(win=5)(x, x) < -0.99
    combo = torch.rand(1, 5, 1, 16, 16)
    combo[:, -1] = 1.0
    assert MaskedNCC(win=3, n_channels=4)(combo, combo) < -0.99
    assert smoothness_loss(torch.zeros(1, 2, 1, 8, 8)) == 0


def test_flow_layouts():
    f = np.random.rand(2, 8, 9).astype(np.float32)
    assert ensure_flow_2hw(f).shape == (2, 8, 9)
    assert np.array_equal(ensure_flow_2hw(np.moveaxis(f, 0, -1)), f)
    assert np.array_equal(ensure_flow_2hw(np.concatenate([np.zeros((1, 8, 9), np.float32), f])), f)
    assert ensure_flow_2hw(torch.from_numpy(f)[None, :, None]).shape == (2, 8, 9)


def test_inverse_warp_identity_and_translation(rng):
    pts = rng.uniform(5, 25, size=(50, 2)).astype(np.float32)
    zero = np.zeros((2, 32, 32), np.float32)
    assert np.abs(find_inverse_warp_coords(pts, zero, device="cpu") - np.round(pts)).max() <= 0.5 + 1e-6
    shift = np.zeros((2, 32, 32), np.float32)
    shift[1] = 3.0  # fixed(p) = moving(p + (0, 3)) => a moving point at x maps to x - 3
    moved = find_inverse_warp_coords(pts, shift, device="cpu")
    assert np.abs(moved[:, 0] - np.round(pts[:, 0] - 3)).max() <= 0.5 + 1e-6
    half = transport_points(pts, shift, alpha=0.5, device="cpu")
    assert np.abs(half[:, 0] - (pts[:, 0] - 1.5)).max() <= 1.0


def test_upsample_and_warp_image():
    f = np.ones((2, 16, 16), np.float32)
    up = upsample_flow(f, (64, 64))
    assert up.shape == (2, 64, 64) and np.allclose(up, 63 / 15)
    # warp_image follows the sampling convention of the network's SpatialTransformer,
    # so images warped with a predicted field match the network's own warped output.
    from axis.registration.blocks import SpatialTransformer_block

    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float32)
    field = rng.normal(0, 2, (2, 32, 32)).astype(np.float32)
    out = warp_image(img, field)
    src = torch.from_numpy(img).permute(2, 0, 1)[None, :, None]
    flow3 = torch.from_numpy(np.concatenate([np.zeros((1, 32, 32), np.float32), field]))[None, :, None]
    ref = SpatialTransformer_block()(src, flow3)[0, :, 0].permute(1, 2, 0).numpy()
    assert np.allclose(out, ref, atol=1e-5)
    assert warp_image((img * 255).astype(np.uint8), field).dtype == np.uint8
