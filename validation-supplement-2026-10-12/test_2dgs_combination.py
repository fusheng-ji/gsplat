"""Integration experiments for PRs #1068, #1070, #1079 and #1080."""

import math

import gsplat
import pytest
import torch

from tests.test_2dgs import _direct_color_scene

SH0 = 0.28209479177387814
MODES = [("RGB", False), ("RGB", True), ("RGB+D", False), ("RGB+ED", False)]
GRAD_KEYS = ("means", "quats", "scales", "opacities", "colors", "viewmats")


def render(scene, packed, mode, median):
    return gsplat.rasterization_2dgs(
        **scene,
        width=32,
        height=24,
        packed=packed,
        render_mode=mode,
        compute_median=median,
        distloss=mode != "RGB",
    )


def independent_sh(scene, per_view, mode, median):
    batch_dims = scene["means"].shape[:-2]
    B, N = math.prod(batch_dims), scene["means"].shape[-2]
    C = scene["viewmats"].shape[-3]
    flat = {
        key: value.reshape(B, *value.shape[len(batch_dims) :])
        for key, value in scene.items()
    }
    images = []
    for batch in range(B):
        for camera in range(C):
            rgb = flat["colors"][batch]
            if per_view:
                rgb = rgb[camera]
            images.append(
                gsplat.rasterization_2dgs(
                    **{
                        key: flat[key][batch]
                        for key in ("means", "quats", "scales", "opacities")
                    },
                    colors=((rgb - 0.5) / SH0).reshape(N, 1, 3),
                    viewmats=flat["viewmats"][batch, camera : camera + 1],
                    Ks=flat["Ks"][batch, camera : camera + 1],
                    width=32,
                    height=24,
                    packed=False,
                    sh_degree=0,
                    render_mode=mode,
                    compute_median=median,
                    distloss=mode != "RGB",
                )
            )
    return {
        index: torch.stack([image[index][0] for image in images]).reshape(
            *batch_dims, C, *images[0][index].shape[1:]
        )
        for index in (0, 1, 2, 4, 5)
    }


@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("per_view", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("clipping", ["none", "partial", "all"])
@pytest.mark.parametrize("mode,median", MODES)
def test_distinct_colors(batch_dims, cameras, per_view, packed, clipping, mode, median):
    scene = _direct_color_scene(batch_dims, cameras, per_view, clipping)
    oracle = {
        key: value.detach().clone().requires_grad_(value.requires_grad)
        for key, value in scene.items()
    }
    actual = render(scene, packed, mode, median)
    expected = independent_sh(oracle, per_view, mode, median)
    for index, reference in expected.items():
        assert actual[index].shape == reference.shape
        assert actual[index].dtype == reference.dtype
        assert actual[index].device == reference.device
        assert torch.isfinite(actual[index]).all()
        torch.testing.assert_close(actual[index], reference, rtol=1e-4, atol=1e-5)
    weights = torch.linspace(0.5, 1.5, actual[0].numel(), device="cuda").reshape_as(
        actual[0]
    )
    (actual[0] * weights).mean().backward()
    (expected[0] * weights).mean().backward()
    for key in GRAD_KEYS:
        assert scene[key].grad is not None
        assert scene[key].grad.shape == scene[key].shape
        assert torch.isfinite(scene[key].grad).all()
        # Upstream's batched geometry backward differs from per-view rendering.
        # Compare that unchanged path to batched SH separately below.
        if not batch_dims or key in ("colors", "opacities"):
            torch.testing.assert_close(
                scene[key].grad, oracle[key].grad, rtol=1e-4, atol=1e-5
            )
    culled = {"none": [], "partial": [1], "all": [0, 1, 2]}[clipping]
    assert torch.count_nonzero(scene["colors"].grad[..., culled, :]) == 0


@pytest.mark.parametrize("batch_dims", [(2,), (1, 2)])
@pytest.mark.parametrize("per_view", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("mode,median", MODES)
def test_batched_geometry(batch_dims, per_view, packed, mode, median):
    scene = _direct_color_scene(batch_dims, 2, per_view, "partial")
    palette = scene["colors"].detach().reshape(-1, 3, 3)[0]
    scene["colors"] = palette.expand_as(scene["colors"]).clone().requires_grad_()
    oracle = {
        key: value.detach().clone().requires_grad_(value.requires_grad)
        for key, value in scene.items()
    }
    oracle["colors"] = ((palette - 0.5) / SH0).unsqueeze(1).requires_grad_()
    actual = render(scene, packed, mode, median)[0]
    expected = gsplat.rasterization_2dgs(
        **oracle,
        width=32,
        height=24,
        packed=packed,
        sh_degree=0,
        render_mode=mode,
        compute_median=median,
        distloss=mode != "RGB",
    )[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    weights = torch.linspace(0.5, 1.5, actual.numel(), device="cuda").reshape_as(actual)
    (actual * weights).mean().backward()
    (expected * weights).mean().backward()
    for key in ("means", "quats", "scales", "opacities", "viewmats"):
        assert torch.isfinite(scene[key].grad).all()
        torch.testing.assert_close(
            scene[key].grad, oracle[key].grad, rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(
        scene["colors"].grad.reshape(-1, 3, 3).sum(0),
        oracle["colors"].grad[:, 0] / SH0,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("per_view", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_empty_median(batch_dims, cameras, per_view, packed, background):
    scene = _direct_color_scene(batch_dims, cameras, per_view, "none")
    for key in ("means", "quats", "scales", "opacities", "colors"):
        axis = -1 if key == "opacities" else -2
        scene[key] = scene[key].detach().narrow(axis, 0, 0).requires_grad_()
    bg = (
        torch.rand(*batch_dims, cameras, 3, device="cuda", requires_grad=True)
        if background
        else None
    )
    result = gsplat.rasterization_2dgs(
        **scene,
        width=32,
        height=24,
        packed=packed,
        backgrounds=bg,
        compute_median=True,
    )
    expected = (
        bg[..., None, None, :].expand_as(result[0])
        if bg is not None
        else torch.zeros_like(result[0])
    )
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
    assert result[3] is None  # RGB mode has no surface normals from depth.
    for index in (0, 1, 2, 4, 5):
        assert torch.isfinite(result[index]).all()
    for index in (1, 2, 4, 5):
        assert torch.count_nonzero(result[index]) == 0
    (result[0].sum() + result[5].sum()).backward()
    for key in GRAD_KEYS:
        assert scene[key].grad is not None
        assert scene[key].grad.shape == scene[key].shape
        assert torch.isfinite(scene[key].grad).all()
        assert torch.count_nonzero(scene[key].grad) == 0
    if bg is not None:
        torch.testing.assert_close(
            bg.grad, torch.full_like(bg, 32 * 24), rtol=0, atol=0
        )
