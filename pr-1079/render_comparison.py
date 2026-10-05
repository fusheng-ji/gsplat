"""PR #1079 measured rendering evidence (CUDA required).

Run once in each checkout:
  python render_comparison.py --version before --output results
  python render_comparison.py --version after --output results
Then plot/compare both runs:
  python render_comparison.py --summarize --output results

Optional --extension loads a separately built gsplat_cuda.so, bypassing JIT.
Each render runs in a subprocess; a SIGFPE is recorded, never drawn as a frame.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys


def worker(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import numpy as np
    import torch

    if args.extension:
        import importlib.util

        spec = importlib.util.spec_from_file_location("gsplat_cuda", args.extension)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules["gsplat.csrc"] = native
    import gsplat

    torch.manual_seed(1079)
    torch.set_default_device("cuda:0")
    empty = args.scene == "empty"
    width, height = (128, 96) if empty else (192, 144)
    cameras = 2 if empty else 1
    if empty:
        means = torch.empty(0, 3)
        quats = torch.empty(0, 4)
        scales = torch.empty(0, 3)
        opacities = torch.empty(0)
        colors = torch.empty(0, 1, 3)
    else:
        yy, xx = torch.meshgrid(
            torch.linspace(-0.8, 0.8, 28), torch.linspace(-1.1, 1.1, 36), indexing="ij"
        )
        zz = 3 + 0.32 * torch.sin(3 * xx) * torch.cos(4 * yy)
        means = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(len(means), 1)
        scales = torch.tensor([0.041, 0.041, 0.025]).repeat(len(means), 1)
        opacities = torch.full((len(means),), 0.8)
        rgb = torch.stack(
            [
                0.5 + 0.45 * torch.sin(2.5 * xx),
                0.5 + 0.45 * torch.sin(3 * yy + 1),
                0.5 + 0.45 * torch.cos(2 * xx - 3 * yy),
            ],
            -1,
        ).reshape(-1, 3)
        colors = ((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)
    inputs = dict(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors
    )
    for tensor in inputs.values():
        tensor.requires_grad_()
    viewmats = torch.eye(4).repeat(cameras, 1, 1).requires_grad_()
    Ks = torch.tensor(
        [[150.0, 0.0, width / 2], [0.0, 150.0, height / 2], [0.0, 0.0, 1.0]]
    ).repeat(cameras, 1, 1)
    background = (
        torch.tensor([[0.10, 0.65, 0.75], [0.85, 0.45, 0.10]], requires_grad=True)
        if empty
        else None
    )
    render = (
        gsplat.rasterization if args.renderer == "3dgs" else gsplat.rasterization_2dgs
    )
    kwargs = dict(
        **inputs,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=bool(args.packed),
        sh_degree=0,
        backgrounds=background,
    )
    if args.renderer == "2dgs":
        kwargs["distloss"] = True
    result = render(**kwargs, render_mode="RGB+ED")
    image, alpha, meta = result[0], result[1], result[-1]
    arrays = dict(
        rgb=image[..., :3].detach().cpu().numpy(),
        alpha=alpha.detach().cpu().numpy(),
        depth=image[..., 3:].detach().cpu().numpy(),
    )
    finite = all(np.isfinite(value).all() for value in arrays.values())
    details = dict(
        renderer=args.renderer,
        packed=bool(args.packed),
        scene=args.scene,
        version=args.version,
        n_gaussians=len(means),
        cameras=cameras,
        width=width,
        height=height,
        dtype=str(image.dtype),
        device=str(image.device),
        finite=bool(finite),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        metadata_shapes={
            key: list(meta[key].shape)
            for key in (
                "radii",
                "means2d",
                "depths",
                "tiles_per_gauss",
                "isect_ids",
                "flatten_ids",
                "isect_offsets",
            )
        },
    )
    if empty:
        expected = background.detach().cpu().numpy()[:, None, None, :]
        details.update(
            rgb_background_max_error=float(np.max(np.abs(arrays["rgb"] - expected))),
            alpha_max_abs=float(np.abs(arrays["alpha"]).max()),
            depth_max_abs=float(np.abs(arrays["depth"]).max()),
        )
        if args.renderer == "2dgs":
            details["auxiliary_max_abs"] = {
                name: float(tensor.detach().abs().max())
                for name, tensor in zip(
                    ("normals", "normals_from_depth", "distortion", "median"),
                    result[2:6],
                )
                if tensor is not None
            }
        image[..., :3].sum().backward()
        details["background_gradient"] = background.grad.cpu().tolist()
        details["background_gradient_max_error"] = float(
            (background.grad - width * height).abs().max()
        )
        details["empty_gradient_shapes"] = {
            key: list(tensor.grad.shape) if tensor.grad is not None else None
            for key, tensor in inputs.items()
        }
        assert (
            finite
            and details["rgb_background_max_error"] == 0
            and details["alpha_max_abs"] == 0
            and details["depth_max_abs"] == 0
        )
        assert details["background_gradient_max_error"] == 0
        assert all(
            tensor.grad is not None
            and tensor.grad.shape == tensor.shape
            and tensor.grad.numel() == 0
            for tensor in inputs.values()
        )
        assert all(
            value == 0 for value in details.get("auxiliary_max_abs", {}).values()
        )
    else:
        # RGB-only loss avoids a separate, pre-existing 2DGS depth-backward issue.
        if args.renderer == "2dgs":
            kwargs["distloss"] = False
        rgb_result = render(**kwargs, render_mode="RGB")[0]
        weight = torch.linspace(0.5, 1.5, height * width).reshape(1, height, width, 1)
        loss = (rgb_result * weight).mean()
        loss.backward()
        details["rgb_loss"] = float(loss.detach())
        details[
            "gradient_loss"
        ] = "mean(RGB * linspace(0.5, 1.5, H*W).reshape(1,H,W,1))"
        for key, tensor in inputs.items():
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            arrays["grad_" + key] = tensor.grad.detach().cpu().numpy()
        assert finite and np.count_nonzero(arrays["alpha"]) > 0
    prefix = (
        Path(args.output)
        / f"{args.version}-{args.scene}-{args.renderer}-packed{args.packed}"
    )
    np.savez_compressed(str(prefix) + ".npz", **arrays)
    prefix.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")


def collect(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for scene in ("empty", "control"):
        for renderer in ("3dgs", "2dgs"):
            for packed in (0, 1):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--version",
                    args.version,
                    "--scene",
                    scene,
                    "--renderer",
                    renderer,
                    "--packed",
                    str(packed),
                    "--output",
                    str(output.resolve()),
                ]
                if args.extension:
                    command += ["--extension", str(Path(args.extension).resolve())]
                try:
                    child = subprocess.run(
                        command, capture_output=True, text=True, timeout=180
                    )
                    status = dict(
                        version=args.version,
                        scene=scene,
                        renderer=renderer,
                        packed=bool(packed),
                        returncode=child.returncode,
                        signal=signal.Signals(-child.returncode).name
                        if child.returncode < 0
                        else None,
                        stdout=child.stdout,
                        stderr=child.stderr,
                    )
                except subprocess.TimeoutExpired:
                    status = dict(
                        version=args.version,
                        scene=scene,
                        renderer=renderer,
                        packed=bool(packed),
                        timeout_seconds=180,
                    )
                prefix = output / f"{args.version}-{scene}-{renderer}-packed{packed}"
                prefix.with_suffix(".status.json").write_text(
                    json.dumps(status, indent=2) + "\n"
                )
                print(
                    args.version,
                    scene,
                    renderer,
                    packed,
                    status.get("returncode", "timeout"),
                    flush=True,
                )
                expected = (
                    -signal.SIGFPE
                    if args.version == "before" and scene == "empty"
                    else 0
                )
                assert status.get("returncode") == expected, status


def summarize(args):
    import csv
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(args.output)
    configs = [(renderer, packed) for renderer in ("3dgs", "2dgs") for packed in (0, 1)]
    rows = []
    for renderer, packed in configs:
        prefix = f"empty-{renderer}-packed{packed}"
        empty = json.loads((output / f"after-{prefix}.json").read_text())
        before_status = json.loads(
            (output / f"before-{prefix}.status.json").read_text()
        )
        before = np.load(output / f"before-control-{renderer}-packed{packed}.npz")
        after = np.load(output / f"after-control-{renderer}-packed{packed}.npz")
        row = dict(
            renderer=renderer,
            packed=bool(packed),
            before_empty_signal=before_status["signal"],
            after_background_error=empty["rgb_background_max_error"],
            after_alpha_max=empty["alpha_max_abs"],
            after_depth_max=empty["depth_max_abs"],
            background_gradient_error=empty["background_gradient_max_error"],
            control_rgb_max_delta=float(np.abs(after["rgb"] - before["rgb"]).max()),
            control_alpha_max_delta=float(
                np.abs(after["alpha"] - before["alpha"]).max()
            ),
            control_depth_max_delta=float(
                np.abs(after["depth"] - before["depth"]).max()
            ),
            control_gradient_max_delta=max(
                float(np.abs(after[key] - before[key]).max())
                for key in before.files
                if key.startswith("grad_")
            ),
            control_gradient_max_deltas={
                key: float(np.abs(after[key] - before[key]).max())
                for key in before.files
                if key.startswith("grad_")
            },
        )
        rows.append(row)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in rows[0] if key != "control_gradient_max_deltas"],
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in row.items()
                if key != "control_gradient_max_deltas"
            }
            for row in rows
        )
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
    fig, axes = plt.subplots(4, 4, figsize=(14, 7.5), constrained_layout=True)
    for i, (renderer, packed) in enumerate(configs):
        data = np.load(output / f"after-empty-{renderer}-packed{packed}.npz")
        axes[i, 0].set_facecolor("#f3f3f3")
        axes[i, 0].text(
            0.5,
            0.5,
            "SIGFPE\nNo rendered frame",
            color="#a32f2f",
            ha="center",
            va="center",
            transform=axes[i, 0].transAxes,
        )
        for j, key in enumerate(("rgb", "alpha", "depth"), 1):
            value = np.concatenate(list(data[key]), axis=1)
            axes[i, j].imshow(
                value if key == "rgb" else value[..., 0],
                vmin=0,
                vmax=1 if key != "depth" else 5,
                cmap="gray" if key != "rgb" else None,
            )
            axes[i, j].set_aspect("auto")
        axes[i, 0].set_ylabel(
            f"{renderer.upper()}\n{'packed' if packed else 'unpacked'}"
        )
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(
        axes[0],
        (
            "Before",
            "After RGB: camera 0 | camera 1",
            "After alpha [0,1]",
            "After ED depth [0,5]",
        ),
    ):
        ax.set_title(title)
    fig.suptitle(
        "Empty scene: 512d366 → 065f4a8 | N=0, 2 cameras, 128×96, RGB+ED, float32\nBefore has no output. After: supplied backgrounds; alpha and depth exactly zero.",
        fontsize=13,
    )
    fig.savefig(
        output / "empty-scene.png", dpi=160, bbox_inches="tight", pad_inches=0.15
    )
    plt.close(fig)
    for channel, filename, title, vmax in (
        ("rgb", "control-rgb.png", "RGB", 1),
        ("depth", "control-depth.png", "expected depth", 4),
    ):
        fig, axes = plt.subplots(
            4,
            3,
            figsize=(12.5 if channel == "depth" else 11, 10),
            constrained_layout=True,
        )
        max_delta = max(row[f"control_{channel}_max_delta"] for row in rows)
        # Fixed epsilon floor prevents a zero error plot from autoscaling noise.
        error_scale = max(max_delta, 1e-6)
        for i, (renderer, packed) in enumerate(configs):
            before = np.load(output / f"before-control-{renderer}-packed{packed}.npz")[
                channel
            ][0]
            after = np.load(output / f"after-control-{renderer}-packed{packed}.npz")[
                channel
            ][0]
            for j, value in enumerate((before, after)):
                source_plot = axes[i, j].imshow(
                    value if channel == "rgb" else value[..., 0],
                    vmin=0,
                    vmax=vmax,
                    cmap=None if channel == "rgb" else "viridis",
                )
            error = np.abs(after - before).max(axis=-1)
            plotted = axes[i, 2].imshow(error, vmin=0, vmax=error_scale, cmap="magma")
            axes[i, 0].set_ylabel(
                f"{renderer.upper()}\n{'packed' if packed else 'unpacked'}"
            )
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, label in zip(
            axes[0],
            (
                "Before: 512d366",
                "After: 065f4a8",
                f"Absolute error (max={max_delta:.2e})",
            ),
        ):
            ax.set_title(label)
        if channel == "depth":
            fig.colorbar(
                source_plot,
                ax=axes[:, :2],
                shrink=0.6,
                label="expected depth (scene units)",
            )
        fig.colorbar(plotted, ax=axes[:, 2], shrink=0.6, label="absolute error")
        fig.suptitle(
            f"Nonempty control: {title} | N=1008, 1 camera, 192×144, SH degree 0\nRGB+ED forward; no supplied background. Shared color/depth scales across before and after.",
            fontsize=13,
        )
        fig.savefig(output / filename, dpi=160, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("before", "after"))
    parser.add_argument("--output", default="results")
    parser.add_argument("--extension")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--scene", choices=("empty", "control"))
    parser.add_argument("--renderer", choices=("3dgs", "2dgs"))
    parser.add_argument("--packed", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.summarize:
        summarize(args)
    else:
        parser.error("--version required") if args.version is None else collect(args)
