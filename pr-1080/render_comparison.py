"""Measured PR #1080 evidence; run in each checkout, then --summarize.

python render_comparison.py --version before --output results
python render_comparison.py --version after --output results
python render_comparison.py --summarize --output results
Optional --extension loads an existing native build instead of using JIT.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

CASES = {
    "unpacked_shared": ((), 1, False, False, "none"),
    "packed_partial": ((), 1, True, False, "partial"),
    "packed_per_camera": ((), 2, True, True, "partial"),
    "packed_batched": ((2,), 2, True, True, "partial"),
    "packed_all_culled": ((), 1, True, False, "all"),
}
OUTPUTS = {"image": 0, "alpha": 1, "normals": 2, "distortion": 4, "median": 5}
GRADIENTS = ("means", "quats", "scales", "opacities", "colors", "viewmats")


def worker(args):
    import numpy as np
    import torch

    if args.extension:
        import importlib.util

        spec = importlib.util.spec_from_file_location("gsplat_cuda", args.extension)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules["gsplat.csrc"] = native
    import gsplat

    torch.manual_seed(1080)
    torch.set_default_device("cuda:0")
    batch_dims, cameras, packed, per_view, clipping = CASES[args.case]
    width, height = 192, 144
    yy, xx = torch.meshgrid(
        torch.linspace(-0.8, 0.8, 28), torch.linspace(-1.1, 1.1, 36), indexing="ij"
    )
    zz = 3 + 0.32 * torch.sin(3 * xx) * torch.cos(4 * yy)
    means = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    if clipping == "partial":
        means[1::4, 2] *= -1  # Remove interleaved IDs, not a suffix.
    elif clipping == "all":
        means[:, 2] *= -1
    count = len(means)
    rgb = torch.stack(
        [
            0.5 + 0.45 * torch.sin(2.5 * xx),
            0.5 + 0.45 * torch.sin(3 * yy + 1),
            0.5 + 0.45 * torch.cos(2 * xx - 3 * yy),
        ],
        -1,
    ).reshape(count, 3)
    batch_count = math.prod(batch_dims)
    palettes = []
    for batch in range(batch_count):
        palette = rgb.roll(batch, -1)
        if per_view:
            palette = torch.stack(
                [palette.roll(camera, -1) for camera in range(cameras)]
            )
        palettes.append(palette)
    colors = torch.stack(palettes).reshape(
        batch_dims + ((cameras,) if per_view else ()) + (count, 3)
    )
    scene = {
        "means": means.expand(batch_dims + means.shape).clone(),
        "quats": torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(*batch_dims, count, 1),
        "scales": torch.tensor([0.041, 0.041, 0.025]).repeat(*batch_dims, count, 1),
        "opacities": torch.full(batch_dims + (count,), 0.8),
        "colors": colors,
        "viewmats": torch.eye(4).repeat(*batch_dims, cameras, 1, 1),
        "Ks": torch.tensor(
            [[150.0, 0.0, width / 2], [0.0, 150.0, height / 2], [0.0, 0.0, 1.0]]
        ).repeat(*batch_dims, cameras, 1, 1),
    }
    scene["viewmats"][..., :, 0, 3] = torch.arange(cameras) * 0.12
    for key in GRADIENTS:
        scene[key].requires_grad_()
    reference_scene = {
        key: value.detach().clone().requires_grad_(value.requires_grad)
        for key, value in scene.items()
    }

    def reference(mode):
        flat = {
            key: value.reshape(batch_count, *value.shape[len(batch_dims) :])
            for key, value in reference_scene.items()
        }
        renders = []
        for batch in range(batch_count):
            for camera in range(cameras):
                palette = flat["colors"][batch]
                if per_view:
                    palette = palette[camera]
                renders.append(
                    gsplat.rasterization_2dgs(
                        **{
                            key: flat[key][batch]
                            for key in ("means", "quats", "scales", "opacities")
                        },
                        colors=((palette - 0.5) / 0.28209479177387814).reshape(
                            count, 1, 3
                        ),
                        viewmats=flat["viewmats"][batch, camera : camera + 1],
                        Ks=flat["Ks"][batch, camera : camera + 1],
                        width=width,
                        height=height,
                        packed=False,
                        sh_degree=0,
                        render_mode=mode,
                        distloss=mode != "RGB",
                    )
                )
        return {
            name: torch.stack([render[index][0] for render in renders]).reshape(
                batch_dims + (cameras,) + renders[0][index].shape[1:]
            )
            for name, index in OUTPUTS.items()
        }

    expected = reference("RGB+ED")
    reference_rgb = reference("RGB")["image"]
    weight = torch.linspace(0.5, 1.5, reference_rgb.numel()).reshape_as(reference_rgb)
    (reference_rgb * weight).mean().backward()
    arrays = {
        "reference_" + name: value.detach().cpu().numpy()
        for name, value in expected.items()
    }
    for key in GRADIENTS:
        arrays["reference_grad_" + key] = reference_scene[key].grad.cpu().numpy()
    details = dict(
        case=args.case,
        version=args.version,
        batch_dims=batch_dims,
        cameras=cameras,
        packed=packed,
        per_camera_colors=per_view,
        clipping=clipping,
        n_gaussians=count,
        width=width,
        height=height,
        dtype=str(expected["image"].dtype),
        device=str(expected["image"].device),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        direct_status="success",
    )
    try:
        actual_rgb = gsplat.rasterization_2dgs(
            **scene,
            width=width,
            height=height,
            packed=packed,
            sh_degree=None,
            render_mode="RGB",
            distloss=False,
        )[0]
        rendered = gsplat.rasterization_2dgs(
            **scene,
            width=width,
            height=height,
            packed=packed,
            sh_degree=None,
            render_mode="RGB+ED",
            distloss=True,
        )
    except RuntimeError as error:
        if args.version != "before":
            raise
        details.update(direct_status="RuntimeError", error=str(error))
    else:
        for name, index in OUTPUTS.items():
            torch.testing.assert_close(
                rendered[index], expected[name], rtol=1e-4, atol=1e-5
            )
            arrays["direct_" + name] = rendered[index].detach().cpu().numpy()
        (actual_rgb * weight).mean().backward()
        compared = GRADIENTS if not batch_dims else ("colors", "opacities")
        for key in GRADIENTS:
            arrays["direct_grad_" + key] = scene[key].grad.cpu().numpy()
            assert scene[key].grad.shape == scene[key].shape
            if key in compared:
                torch.testing.assert_close(
                    scene[key].grad, reference_scene[key].grad, rtol=1e-4, atol=1e-5
                )
        details["gradient_reference_keys"] = list(compared)
        details["projected_gaussians"] = (
            int(rendered[-1]["gaussian_ids"].numel()) if packed else count
        )
        if clipping != "none":
            culled = slice(1, None, 4) if clipping == "partial" else slice(None)
            details["culled_color_gradient_max_abs"] = float(
                scene["colors"].grad[..., culled, :].abs().max()
            )
            assert details["culled_color_gradient_max_abs"] == 0
    details["finite"] = bool(all(np.isfinite(value).all() for value in arrays.values()))
    assert details["finite"]
    stem = args.output / f"{args.version}-{args.case}"
    np.savez_compressed(str(stem) + ".npz", **arrays)
    Path(str(stem) + ".json").write_text(json.dumps(details, indent=2) + "\n")
    print(args.version, args.case, details["direct_status"], flush=True)


def summarize(args):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = []
    for case in CASES:
        before = np.load(args.output / f"before-{case}.npz")
        after = np.load(args.output / f"after-{case}.npz")
        details = json.loads((args.output / f"after-{case}.json").read_text())
        record = {
            "case": case,
            "before_status": json.loads(
                (args.output / f"before-{case}.json").read_text()
            )["direct_status"],
        }
        for name in OUTPUTS:
            record[name + "_max_abs_error"] = float(
                np.abs(after["direct_" + name] - after["reference_" + name]).max()
            )
            assert np.array_equal(
                before["reference_" + name], after["reference_" + name]
            ), name
        record["rgb_max_abs_error"] = float(
            np.abs(
                after["direct_image"][..., :3] - after["reference_image"][..., :3]
            ).max()
        )
        record["depth_max_abs_error"] = float(
            np.abs(
                after["direct_image"][..., 3:] - after["reference_image"][..., 3:]
            ).max()
        )
        for key in details["gradient_reference_keys"]:
            record[key + "_gradient_max_abs_error"] = float(
                np.abs(
                    after["direct_grad_" + key] - after["reference_grad_" + key]
                ).max()
            )
        record["culled_color_gradient_max_abs"] = details.get(
            "culled_color_gradient_max_abs", 0.0
        )
        record["projected_gaussians"] = details["projected_gaussians"]
        record["finite"] = details["finite"]
        metrics.append(record)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fields = sorted(set().union(*(record.keys() for record in metrics)))
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    shown = list(CASES)[:4]
    fig, axes = plt.subplots(len(shown), 4, figsize=(15, 10), constrained_layout=True)
    for row, case in enumerate(shown):
        arrays = np.load(args.output / f"after-{case}.npz")
        # Multi-view rows show batch 0 / camera 0; all other views remain in NPZ.
        reference = arrays["reference_image"].reshape(-1, 144, 192, 4)[0, ..., :3]
        actual = arrays["direct_image"].reshape(-1, 144, 192, 4)[0, ..., :3]
        error = np.abs(actual - reference).max(-1)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_facecolor("#f3f3f3")
        axes[row, 0].text(
            0.5,
            0.5,
            "RuntimeError\nNo rendered frame",
            ha="center",
            va="center",
            transform=axes[row, 0].transAxes,
            color="#a12828",
        )
        axes[row, 0].set_ylabel(case.replace("_", "\n"), fontsize=10)
        axes[row, 1].imshow(reference, vmin=0, vmax=1)
        axes[row, 2].imshow(actual, vmin=0, vmax=1)
        plot = axes[row, 3].imshow(error, cmap="magma", vmin=0, vmax=1e-6)
        axes[row, 3].set_xlabel(f"shown-view max error: {error.max():.2e}", fontsize=9)
    for ax, title in zip(
        axes[0],
        [
            "Before: direct RGB",
            "SH degree 0 reference",
            "After: direct RGB",
            "Absolute RGB error (max channel)",
        ],
    ):
        ax.set_title(title, fontsize=11)
    fig.colorbar(
        plot, ax=axes[:, 3], shrink=0.6, label="absolute error; fixed scale [0, 1e-6]"
    )
    fig.suptitle(
        "PR #1080 | 512d366 → 8bf4b76 | float32 / B200 / 192×144\nInterleaved culling; multi-view rows show batch 0, camera 0",
        fontsize=13,
    )
    fig.savefig(
        args.output / "rgb-comparison.png",
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)

    arrays = np.load(args.output / "after-packed_batched.npz")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for batch in range(2):
        for camera in range(2):
            for offset, prefix in enumerate(("reference_", "direct_")):
                ax = axes[batch, 2 * camera + offset]
                ax.imshow(
                    arrays[prefix + "image"][batch, camera, ..., :3], vmin=0, vmax=1
                )
                ax.set_title(
                    f"B{batch} / C{camera}: "
                    + ("SH reference" if offset == 0 else "direct RGB")
                )
                ax.axis("off")
    fig.suptitle(
        "Distinct batch / camera colors | packed, 1008 Gaussians, 756 visible per view",
        fontsize=13,
    )
    fig.savefig(
        args.output / "batch-camera-comparison.png",
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("before", "after"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args)
    elif args.case:
        worker(args)
    else:
        if args.version is None:
            parser.error("--version is required for rendering")
        for case in CASES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--version",
                args.version,
                "--output",
                str(args.output.resolve()),
                "--case",
                case,
            ]
            if args.extension:
                command.extend(["--extension", args.extension])
            subprocess.run(command, check=True, timeout=120)


if __name__ == "__main__":
    main()
