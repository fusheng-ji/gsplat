"""#978 evidence: gradient error vs. reference, and depth-only optimization.

Run once per source tree (main / fix); the CUDA backend comes from whichever
gsplat is on PYTHONPATH. The PyTorch reference is independent of the kernel.
"""
import json, math, os, sys, time
SEED = int(os.environ.get("SEED", "0"))
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import (fully_fused_projection_with_ut, isect_tiles,
    isect_offset_encode, rasterize_to_pixels_eval3d_extra)
from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
import gsplat.cuda._torch_impl_eval3d as _ref

# Count fragments (Gaussian x pixel) whose alpha hits MAX_ALPHA, inside the reference.
FRAG = {"clamped": 0, "valid": 0}
_orig_alphas = _ref._compute_gaussian_alphas
def _counting_alphas(grayDist, opac, *a, **k):
    alphas, resp = _orig_alphas(grayDist, opac, *a, **k)
    valid = alphas >= 1.0 / 255.0
    FRAG["valid"] += int(valid.sum())
    FRAG["clamped"] += int(((alphas >= 0.99 - 1e-7) & valid).sum())
    return alphas, resp
_ref._compute_gaussian_alphas = _counting_alphas

tag, out_dir = sys.argv[1], sys.argv[2]
run_reference = len(sys.argv) > 3 and sys.argv[3] == "ref"
dev = "cuda"
W = H = 64
TS = 8
Ks = torch.tensor([[[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]]], device=dev)
viewmats = torch.eye(4, device=dev)[None]


def make_scene(seed=0, n=48):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # A tilted, surfel-like surface (thin along one axis), like a road patch.
    uv = torch.rand(n, 2, generator=g) * 2 - 1
    means = torch.stack([uv[:, 0] * 1.6, uv[:, 1] * 1.6, 4.0 + 0.8 * uv[:, 1]], -1)
    scales = torch.stack([torch.full((n,), 0.35), torch.full((n,), 0.35), torch.full((n,), 0.03)], -1)
    scales = scales * (0.8 + 0.4 * torch.rand(n, 3, generator=g))
    quats = F.normalize(torch.tensor([1.0, 0.25, 0.0, 0.0]) + 0.1 * torch.randn(n, 4, generator=g), dim=-1)
    return {k: v.to(dev) for k, v in dict(means=means, quats=quats, scales=scales).items()}


def render(p, opacity, impl):
    n = p["means"].shape[0]
    opac = torch.full((1, n), opacity, device=dev)
    colors = torch.full((1, n, 3), 0.5, device=dev)
    with torch.no_grad():
        radii, m2d, depths, _, _ = fully_fused_projection_with_ut(
            p["means"], p["quats"], p["scales"], opac[0], viewmats, Ks, W, H)
    tw, th = math.ceil(W / TS), math.ceil(H / TS)
    _, ids, fids = isect_tiles(m2d, radii, depths, TS, tw, th)
    offs = isect_offset_encode(ids, 1, tw, th).reshape(1, th, tw)
    args = (p["means"], p["quats"], p["scales"], colors, opac, viewmats, Ks, W, H)
    if impl == "cuda":
        out, alphas, *_ = rasterize_to_pixels_eval3d_extra(
            *args, TS, offs, fids, use_hit_distance=True)
    else:
        out, alphas, *_ = _rasterize_to_pixels_eval3d(
            *args, tile_size=TS, isect_offsets=offs, flatten_ids=fids, use_hit_distance=True)
    return out[0, ..., -1], alphas[0, ..., 0]


result = {"tag": tag}

# 1. Gradient error vs. reference over opacity.
scene = make_scene()
wts = torch.linspace(0.5, 1.5, W * H, device=dev).reshape(H, W)
rows = []
for opacity in (0.5, 0.9, 0.99, 0.995, 0.999):
    grads = {}
    FRAG.update(clamped=0, valid=0)
    for impl in ("cuda", "torch"):
        p = {k: v.clone().requires_grad_(True) for k, v in scene.items()}
        depth, alphas = render(p, opacity, impl)
        (depth * wts).sum().backward()
        grads[impl] = {k: v.grad for k, v in p.items()}
    row = {"opacity": opacity, "clamped_frac": FRAG["clamped"] / FRAG["valid"]}
    FRAG.update(clamped=0, valid=0)
    for k in scene:
        a, b = grads["cuda"][k], grads["torch"][k]
        row[k] = float((a - b).norm() / b.norm())
    rows.append(row)
    print(tag, row, flush=True)
result["grad_error"] = rows

# 2. Depth-only optimization from a perturbed start.
OPACITY, STEPS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.999, int(sys.argv[5]) if len(sys.argv) > 5 else 1000
gt = make_scene(SEED)
with torch.no_grad():
    target, gt_alpha = render(gt, OPACITY, "cuda")
g = torch.Generator(device="cpu").manual_seed(1000 + SEED)
init = {
    "means": gt["means"] + 0.12 * torch.randn(gt["means"].shape, generator=g).to(dev),
    "quats": F.normalize(gt["quats"] + 0.15 * torch.randn(gt["quats"].shape, generator=g).to(dev), dim=-1),
    "log_scales": torch.log(gt["scales"] * 1.3),
}
mask = gt_alpha > 0


def optimize(impl):
    q = {k: v.clone().requires_grad_(True) for k, v in init.items()}
    opt = torch.optim.Adam(q.values(), lr=0.01)
    hist = []
    t0 = time.time()
    for step in range(STEPS + 1):
        p = {"means": q["means"], "quats": F.normalize(q["quats"], dim=-1), "scales": q["log_scales"].exp()}
        depth, _ = render(p, OPACITY, impl)
        loss = F.mse_loss(depth[mask], target[mask])
        if step % 10 == 0 or step == STEPS:
            with torch.no_grad():
                hist.append({
                    "step": step,
                    "depth_rmse": float(((depth - target)[mask] ** 2).mean().sqrt()),
                    "mean_err": float((p["means"] - gt["means"]).norm(dim=-1).mean()),
                    "scale_ratio": float((p["scales"] / gt["scales"]).mean()),
                    "scale_ratio_max": float((p["scales"] / gt["scales"]).max()),
                })
        if step == STEPS:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(tag, impl, hist[0], hist[-1], f"{time.time() - t0:.1f}s", flush=True)
    final_params = {k: v.detach().cpu() for k, v in p.items()}
    return hist, depth.detach().cpu().numpy(), final_params


hist, final, fp = optimize("cuda")
torch.save({"gt": {k: v.cpu() for k, v in gt.items()}, "final": fp}, f"{out_dir}/{tag}_s{SEED}_params.pt")
result["opt_cuda"] = hist
arrays = {"target": target.cpu().numpy(), "final_cuda": final}
with torch.no_grad():
    p0 = {"means": init["means"], "quats": init["quats"], "scales": init["log_scales"].exp()}
    arrays["init"] = render(p0, OPACITY, "cuda")[0].cpu().numpy()
if run_reference:
    hist, final, fp = optimize("torch")
    torch.save(fp, f"{out_dir}/reference_s{SEED}_params.pt")
    result["opt_reference"] = hist
    arrays["final_reference"] = final

result["seed"] = SEED
json.dump(result, open(f"{out_dir}/{tag}_s{SEED}.json", "w"), indent=1)
np.savez(f"{out_dir}/{tag}_s{SEED}.npz", **arrays)
