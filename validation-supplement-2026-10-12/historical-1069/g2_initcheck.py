# Approximates the #1031 setup: spinning lidar with 64 rows x 1800 columns,
# with_ut + with_eval3d, render_mode="RGB-Ed", tiling from compute_lidar_tiling() defaults.
import math, sys, torch, gsplat
from gsplat.cuda._wrapper import SpinningDirection
d = torch.device("cuda:0"); torch.manual_seed(0)
n_rows, n_cols = 64, 1800
params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
    row_elevations_rad=torch.linspace(math.radians(2.0), math.radians(-24.8), n_rows, device=d),
    column_azimuths_rad=torch.linspace(math.pi, -math.pi, n_cols + 1, device=d)[:-1],
    row_azimuth_offsets_rad=(torch.rand(n_rows, device=d) - 0.5) * 0.02,
    spinning_frequency_hz=10.0,
    spinning_direction=SpinningDirection.CLOCKWISE,
)
tiling = gsplat.compute_lidar_tiling(params)  # library defaults of the checked-out revision
lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
    params, gsplat.compute_lidar_angles_to_columns_map(params), tiling)
cnt = tiling.tiles_pack_info[:, 1].long()
print(f"tiling: max elements/tile={cnt.max().item()}, unwritten at tile_size=8: "
      f"{(cnt - 64).clamp_min(0).sum().item() / cnt.sum().item():.1%}")
N = 2000
dirs = torch.nn.functional.normalize(torch.randn(N, 3, device=d), dim=-1)
means = dirs * (torch.rand(N, 1, device=d) * 15 + 5)
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=d), dim=-1)
scales = torch.rand(N, 3, device=d) * 0.3 + 0.05
opac = torch.rand(N, device=d); cols = torch.rand(N, 3, device=d)
def render():
    pass
    r, a, _ = gsplat.rasterization(means, quats, scales, opac, cols, torch.eye(4, device=d)[None],
        torch.eye(3, device=d)[None], n_cols, n_rows, packed=False, with_ut=True, with_eval3d=True,
        camera_model="lidar", lidar_coeffs=lidar, render_mode="RGB-Ed")
    return r.clone(), a.clone()
try:
    (r1, a1) = (r2, a2) = render()
except ValueError as e:
    print("rasterization raised ValueError:", e); sys.exit()
ed = r1[..., 3]
print(f"alpha max={a1.max().item():.4g}, frac alpha>1={(a1 > 1).float().mean().item():.2%}, "
      f"frac alpha<0={(a1 < 0).float().mean().item():.2%}, "
      f"frac Ed not finite={(~torch.isfinite(ed)).float().mean().item():.2%}, "
      f"repeat identical={torch.equal(r1.nan_to_num(), r2.nan_to_num()) and torch.equal(a1, a2)}")
