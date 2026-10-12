"""2DGS rasterization timing for one gsplat build. Prints one JSON line per config."""
import json, sys, torch
import gsplat
from gsplat.rendering import rasterization_2dgs

tag, rnd = sys.argv[1], int(sys.argv[2])
# Optional 3rd arg "median_on": pass compute_median=True (only builds that have the argument).
extra = {"compute_median": True} if len(sys.argv) > 3 and sys.argv[3] == "median_on" else {}
d = "cuda"
W, H = 1920, 1080
Ks = torch.tensor([[1500.0, 0, W / 2], [0, 1500, H / 2], [0, 0, 1]], device=d)[None]
viewmats = torch.eye(4, device=d)[None]

def scene(N):
    g = torch.Generator(device=d).manual_seed(0)
    means = torch.randn(N, 3, device=d, generator=g) * 2
    means[:, 2] = means[:, 2].abs() + 4
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=d, generator=g), dim=-1)
    scales = torch.rand(N, 3, device=d, generator=g) * 0.05
    opac = torch.rand(N, device=d, generator=g)
    colors = torch.randn(N, 16, 3, device=d, generator=g)
    return [t.requires_grad_(True) for t in (means, quats, scales, opac, colors)]

CONFIGS = [  # name, N, render_mode, backward
    ("RGB_fwdbwd_100k", 100_000, "RGB", True),
    ("RGB_fwd_100k", 100_000, "RGB", False),
    ("RGB_fwdbwd_1M", 1_000_000, "RGB", True),
    ("RGBED_fwd_100k", 100_000, "RGB+ED", False),
]
for name, N, mode, bwd in CONFIGS:
    params = scene(N)
    def step():
        out = rasterization_2dgs(*params, viewmats, Ks, W, H, sh_degree=3, render_mode=mode, **extra)
        if bwd:
            out[0].sum().backward()
            for p in params:
                p.grad = None
    ctx = torch.enable_grad() if bwd else torch.no_grad()
    with ctx:
        for _ in range(20):
            step()
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); step(); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    times.sort()
    print(json.dumps({"tag": tag, "round": rnd, "config": name, "median_ms": times[50],
                      "p10_ms": times[10], "p90_ms": times[90], "min_ms": times[0],
                      "gsplat": gsplat.__file__}), flush=True)
