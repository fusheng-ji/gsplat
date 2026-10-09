"""Figures and table for the #978 PR. Forward rendering only (identical in both trees)."""
import glob, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gsplat.rendering import rasterization

X = sys.argv[1]
SEEDS = sorted(int(f.split("_s")[-1][:-5]) for f in glob.glob(f"{X}/fix_s*.json"))
COL = {"main": "#d1495b", "fix": "#2e86ab", "reference": "#444444"}
LAB = {"main": "main", "fix": "this PR", "reference": "PyTorch reference"}


def load(seed):
    m = json.load(open(f"{X}/main_s{seed}.json"))
    f = json.load(open(f"{X}/fix_s{seed}.json"))
    return {"main": m["opt_cuda"], "fix": f["opt_cuda"], "reference": f["opt_reference"]}, m, f


# ---- table ----
rows = []
for s in SEEDS:
    h, _, _ = load(s)
    rows.append({k: h[k][-1] for k in h})
summary = {}
for k in ("main", "fix", "reference"):
    summary[k] = {m: (float(np.mean([r[k][m] for r in rows])), float(np.min([r[k][m] for r in rows])), float(np.max([r[k][m] for r in rows])))
                  for m in ("depth_rmse", "scale_ratio", "scale_ratio_max", "mean_err")}
json.dump({"seeds": SEEDS, "per_seed": rows, "summary": summary}, open(f"{X}/summary.json", "w"), indent=1)
_, m0, f0 = load(SEEDS[0])
json.dump({"main": m0["grad_error"], "fix": f0["grad_error"]}, open(f"{X}/grad_error.json", "w"), indent=1)

# ---- figure 1: scale growth over steps ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for s in SEEDS:
    h, _, _ = load(s)
    for k in ("main", "fix", "reference"):
        steps = [e["step"] for e in h[k]]
        for ax, m in zip(axes, ("scale_ratio_max", "depth_rmse")):
            ax.plot(steps, [e[m] for e in h[k]], color=COL[k], lw=1.4 if s == SEEDS[0] else 0.7,
                    alpha=1.0 if s == SEEDS[0] else 0.45, ls="--" if k == "reference" else "-",
                    label=LAB[k] if s == SEEDS[0] else None)
axes[0].set_yscale("log"); axes[0].set_title("largest scale / ground-truth scale")
axes[1].set_yscale("log"); axes[1].set_title("depth RMSE (training view)")
for ax in axes:
    ax.set_xlabel("optimization step"); ax.grid(alpha=0.3)
axes[0].legend(frameon=False)
fig.suptitle(f"Hit-distance-only optimization, opacity 0.999, {len(SEEDS)} seeds (seed {SEEDS[0]} bold)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{X}/scale-growth.png", dpi=150)

# ---- figure 2: renders ----
dev = "cuda"


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    eye, target, up = map(lambda v: torch.tensor(v, dtype=torch.float32), (eye, target, up))
    z = torch.nn.functional.normalize(target - eye, dim=0)
    x = torch.nn.functional.normalize(torch.cross(z, up, dim=0), dim=0)
    y = torch.cross(z, x, dim=0)
    R = torch.stack([x, y, z])  # world -> camera rows
    vm = torch.eye(4); vm[:3, :3] = R; vm[:3, 3] = -R @ eye
    return vm


def render_rgb(p, viewmat, size=256, focal=230.0):
    n = p["means"].shape[0]
    colors = torch.tensor(plt.cm.tab20(np.arange(n) % 20)[:, :3], dtype=torch.float32, device=dev)
    K = torch.tensor([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], device=dev)
    img, _, _ = rasterization(p["means"].to(dev), p["quats"].to(dev), p["scales"].to(dev),
                              torch.full((n,), 0.999, device=dev), colors, viewmat[None].to(dev), K[None],
                              size, size, backgrounds=torch.ones(1, 3, device=dev))
    return img[0].clamp(0, 1).cpu().numpy()


seed = SEEDS[0]
P = {"main": torch.load(f"{X}/main_s{seed}_params.pt"), "fix": torch.load(f"{X}/fix_s{seed}_params.pt")}
params = {"ground truth": P["fix"]["gt"], "main": P["main"]["final"], "this PR": P["fix"]["final"],
          "PyTorch reference": torch.load(f"{X}/reference_s{seed}_params.pt")}
D = {"main": np.load(f"{X}/main_s{seed}.npz"), "fix": np.load(f"{X}/fix_s{seed}.npz")}
depths = {"ground truth": D["fix"]["target"], "main": D["main"]["final_cuda"], "this PR": D["fix"]["final_cuda"],
          "PyTorch reference": D["fix"]["final_reference"]}
novel = look_at(eye=(0.0, -3.2, 1.2), target=(0.0, 0.0, 4.0))
fig, axes = plt.subplots(2, 4, figsize=(12, 7.0))
vmin, vmax = depths["ground truth"][depths["ground truth"] > 0].min(), depths["ground truth"].max()
for j, name in enumerate(params):
    d = np.where(depths[name] > 0, depths[name], np.nan)
    axes[0, j].imshow(d, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, j].set_title(f"{name}\ntraining-view hit distance", fontsize=9)
    sr = (params[name]["scales"] / params["ground truth"]["scales"]).max().item()
    axes[1, j].imshow(render_rgb(params[name], novel))
    axes[1, j].set_title(f"novel view, one color per Gaussian\nmax scale ratio {sr:.1f}x", fontsize=9)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(h_pad=2.5); fig.savefig(f"{X}/renders.png", dpi=150)
print(json.dumps(summary, indent=1))
