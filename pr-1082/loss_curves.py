"""Plot measured loss versus selected sample count; no training is performed.

python loss_curves.py --repo /path/to/gsplat --output /path/to/pr-1082
The repository must contain both source commits; only PyTorch/NumPy/matplotlib
are required. Loss modules are loaded directly, avoiding CUDA extension builds.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

BEFORE = "512d366b67073d77ca099ede742683c165dfc23b"
AFTER = "2b19a05a3476479e0c33f9392f5d1bd8e07b31fc"
PRED = [1.0, 2.2, 0.9, 2.7, 4.0, 3.4, 5.2, 4.9]
GT = [1.2, 2.4, 0.7, 3.1, 4.5, 2.8, 5.7, 4.2]


def measure(repo):
    functions = {}
    for version, commit in (("before", BEFORE), ("after", AFTER)):
        source = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:gsplat/losses.py"], text=True
        )
        functions[version] = {}
        exec(compile(source, f"{version}_losses.py", "exec"), functions[version])
    records = []
    for device in ("cpu", "cuda"):
        for dtype in (torch.float32, torch.float64):
            for poison in ("nan", "inf", "negative-inf", "overflow"):
                bad = {
                    "nan": float("nan"),
                    "inf": float("inf"),
                    "negative-inf": -float("inf"),
                    "overflow": torch.finfo(dtype).max * 0.75,
                }[poison]
                for name in ("masked_l1", "pearson_depth_loss"):
                    for selected in range(len(PRED) + 1):
                        pair = {}
                        for version in ("before", "after"):
                            pred = torch.tensor(
                                [bad, bad] + PRED,
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            gt = torch.tensor(
                                [0.0, 0.0] + GT,
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            mask = torch.zeros(
                                len(pred), device=device, dtype=torch.bool
                            )
                            mask[2 : 2 + selected] = True
                            loss = functions[version][name](pred, gt, mask)
                            loss.backward()
                            finite = bool(torch.isfinite(loss))
                            assert (
                                pred.grad is not None
                                and torch.isfinite(pred.grad).all()
                            )
                            torch.testing.assert_close(
                                pred.grad[:2],
                                torch.zeros_like(pred.grad[:2]),
                                rtol=0,
                                atol=0,
                            )
                            degenerate = selected < (1 if name == "masked_l1" else 2)
                            if version == "after" or not degenerate:
                                assert finite
                            if degenerate:
                                assert gt.grad is None
                                assert torch.equal(pred.grad, torch.zeros_like(pred))
                                if version == "after":
                                    assert loss.detach().item() == 0
                                else:
                                    assert not finite
                            else:
                                active_pred, active_gt = pred[mask], gt[mask]
                                expected = (
                                    (active_pred - active_gt).abs().mean()
                                    if name == "masked_l1"
                                    else 1
                                    - torch.corrcoef(
                                        torch.stack([active_pred, active_gt])
                                    )[0, 1]
                                )
                                torch.testing.assert_close(
                                    loss, expected, rtol=1e-5, atol=1e-6
                                )
                            pair[version] = (loss.detach(), pred.grad, gt.grad)
                            records.append(
                                dict(
                                    device=device,
                                    dtype=str(dtype),
                                    poison=poison,
                                    function=name,
                                    selected_samples=selected,
                                    version=version,
                                    loss=float(loss.detach()) if finite else None,
                                    finite_loss=finite,
                                    finite_prediction_gradient=True,
                                    prediction_gradient_max_abs=float(
                                        pred.grad.abs().max()
                                    ),
                                    gt_gradient_present=gt.grad is not None,
                                )
                            )
                        if not degenerate:
                            for before, after in zip(pair["before"], pair["after"]):
                                torch.testing.assert_close(
                                    before, after, rtol=0, atol=0
                                )
    return records


def plot(records, output):
    shown = [
        row
        for row in records
        if row["device"] == "cpu"
        and row["dtype"] == "torch.float64"
        and row["poison"] == "nan"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, name in zip(axes, ("masked_l1", "pearson_depth_loss")):
        curves = {
            version: [
                row
                for row in shown
                if row["function"] == name and row["version"] == version
            ]
            for version in ("before", "after")
        }
        for version, color, marker, style, size in (
            ("after", "#2166ac", "o", "-", 7),
            ("before", "#d68000", "s", "--", 4),
        ):
            rows = curves[version]
            ax.plot(
                [row["selected_samples"] for row in rows],
                [np.nan if row["loss"] is None else row["loss"] for row in rows],
                color=color,
                marker=marker,
                linestyle=style,
                markersize=size,
                label=version.capitalize(),
                linewidth=1.8,
            )
        invalid = [
            row["selected_samples"]
            for row in curves["before"]
            if not row["finite_loss"]
        ]
        ax.plot(
            invalid,
            [0.91] * len(invalid),
            "x",
            color="#b2182b",
            markersize=8,
            transform=ax.get_xaxis_transform(),
            linestyle="none",
        )
        ax.annotate(
            "Before: NaN at n=" + ",".join(map(str, invalid)),
            xy=(np.mean(invalid), 0.91),
            xycoords=("data", "axes fraction"),
            xytext=(0.24, 0.87),
            textcoords="axes fraction",
            color="#b2182b",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#b2182b"),
        )
        ax.set_title(name)
        ax.set_xlabel("Selected finite samples (poisoned entries excluded)")
        ax.set_ylabel("Loss")
        ax.set_xticks(range(9))
        ax.set_xlim(-0.35, 8.35)
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="#b2182b", marker="x", linestyle="none"))
        labels.append("NaN status annotation")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Loss versus mask sample count (fixed inputs; no training)\n512d366 → 2b19a05 | CPU float64 shown; CPU/CUDA × float32/64 verified",
        fontsize=12,
    )
    fig.supxlabel(
        "Red × marks a non-finite result and has no numerical y-value; overlapping finite curves are identical.",
        fontsize=9,
    )
    fig.savefig(
        output / "loss-curves.png", dpi=170, bbox_inches="tight", pad_inches=0.15
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = measure(args.repo)
    report = dict(
        before_commit=BEFORE,
        after_commit=AFTER,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        finite_predictions=PRED,
        finite_targets=GT,
        measurements=records,
    )
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    plot(records, args.output)
    print(
        f"{len(records)} measurements; normal losses and gradients match exactly; all after-fix losses finite."
    )


if __name__ == "__main__":
    main()
