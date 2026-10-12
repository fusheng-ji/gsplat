# Supplementary gsplat validation

This folder separates newly measured results (2026-10-12) from recovered
historical records (2026-10-01/02). No dataset, checkpoint, native binary or
temporary Python environment is included. These are local CUDA results, not
GitHub CI results. All PRs use upstream main `512d366` as their base.

## New 2DGS integration experiment

Combine #1079 (`5250312`), #1080 (`8bf4b76`), #1068 (`8b329df`) and #1070
(`3acaba8`) in that order in a separate checkout. All merges were automatic;
the locally tested merge commit is `917f9d0af8c1dc8b48ba7abdf65c7c96335a18bb`.
The merge contains no additional implementation changes.

Result: **486 passed** (114.81 seconds); all repository formatting checks
passed. See `combined-tests.log` and `combined-format-full.log`.

Run `tests/test_2dgs.py`, `tests/test_empty_scene.py` and the attached
`test_2dgs_combination.py`. The extra script defines 368 cases:

- 288 distinct-color cases: batches `()`, `(2,)`, `(1,2)`; two/three cameras;
  shared/per-camera colors; packed/non-packed; no/partial/all clipping;
  RGB with median disabled/enabled, RGB+D and RGB+ED. Partial clipping removes
  the middle Gaussian. All returned images are compared to independent
  single-camera SH-degree-0 renders; RGB/depth losses exercise backward.
- 32 batched geometry cases use batched SH as the geometry-gradient oracle.
  The separate upstream discrepancy between batched and per-view geometry
  backward remains outside these fixes. Colors and opacity still use the
  independent per-view oracle above. No claim is made that the discrepancy is
  fixed.
- 48 empty RGB/median cases cover batches, multiple cameras, packed/non-packed,
  both color layouts and supplied/default backgrounds. Auxiliary outputs are
  zero; gradients retain shape; each background gradient is exactly
  `32 * 24 = 768`.

Image/gradient comparisons use `rtol=1e-4, atol=1e-5`. The existing empty-scene
suite includes timeout-protected subprocesses and two-GPU tests.

Environment: two NVIDIA B200 GPUs, PyTorch 2.11.0+cu128, CUDA 12.8;
`BUILD_3DGS=BUILD_2DGS=BUILD_3DGUT=BUILD_CAMERA_WRAPPERS=1`,
`NUM_CHANNELS=1,3,4,32`, `TORCH_CUDA_ARCH_LIST=10.0`. The extension was built
once and preloaded into pytest and its subprocesses using `prebuilt_loader.py`,
to prevent `pytest.ini` from triggering a build with a different channel list.
Testing dependencies include pytest, pytest-check, pytest-env and nerfacc 0.5.3;
formatting uses Black 22.3.0 and clang-format 22.1.5.

## New Lidar experiment

The #1069 tests add Pandar128 to the all-pixel regression and compare
nontransparent AT128/Pandar128 renders with the PyTorch Eval3D pixel reference,
using each tiling's own projection/intersection metadata. Both the default
64-ray/8x8 tiling and explicit 256-ray/16x16 tiling are exercised. Seed 42,
64 Gaussians, float32, background `[0.25, 0.5, 0.75]`; all outputs are finite
and maximum alpha exceeds 0.73. These comparisons validate pixel accumulation;
they are not an independent projection oracle or a real-data training run.
Result: **6 passed** (32.72 seconds); full formatting passed. See
`lidar-tests.log`, `lidar-format-full.log` and `lidar-metrics.csv`.

Outside visibility-cutoff disagreements, both RGB and alpha use
`rtol=1e-4, atol=1e-5`. Pandar128 has one marginal pixel per tiling: one path
keeps an alpha approximately `1/255`, the other discards it. This is the ray
conversion/alpha-cutoff behavior already documented in
`tests/test_basic.py::test_rasterize_to_pixels_eval3d`. The regression separately
bounds the frequency (at most `1e-5` of pixels), retained alpha, and RGB error
(`1/255 + 1e-5`). Numerical tables include the unfiltered error as well as the
nonboundary error. The initial failed strict comparison is retained in
`lidar-first-reference.log`.

The old Python source (`512d366`) fails the all-pixel/repeat-render regression
for both devices; see `lidar-before.log`. Both old and fixed Python sources
reuse the above native extension. #1069 changes Python tiling/validation only;
the Lidar pixel CUDA kernels are unchanged in this integration build. Therefore
this isolates the tiling change but is not a separate build of main or #1069.
For reproduction, build the extension, put the appropriate gsplat source first
on `PYTHONPATH`, and run `lidar_before.py --source <main-checkout>
--test-file <fixed-checkout>/tests/test_rasterization.py`. For the fixed tests:
`pytest tests/test_rasterization.py -k 'lidar_tiling_fits_tile_size or
lidar_nontransparent_reference' -q -s`.

## Recovered historical evidence

`historical-1068/` contains original logs, raw 30k validation JSONs and the CUDA
event benchmark. The seven-scene training logs explicitly identify `8b329df`,
the current #1068 head, and successful completion. The comparison baseline
`d76c738` already contains the projection backward fix: it is **not** unmodified
main. Main fails at the first garden trainer step; `8b329df` completes 200 steps
(separate original logs and command record included).

The benchmark uses one Slurm-allocated B200, 1920x1080, SH degree 3, seed 0,
20 warmup iterations and 100 CUDA-event samples per configuration/round, three
rounds with rotated version order. `perf/summary.csv` is the median of the three
round medians. Raw records and the original benchmark/Slurm script are included.
They compare main `512d366`, intermediate always-on median `61003a2`, and current
head `8b329df` with median disabled/enabled. These timings were not rerun under
today's shared GPU workload and are not a universal overhead guarantee.

`historical-1069/` contains original initcheck summaries (32,784 errors on main,
zero with the fix), the probe source and retained full-suite logs/failure lists.
The old sanitizer probe's `repeat identical` field compares the same render
tuple to itself; that field provides no repeatability evidence and is ignored.
Today's regression actually performs two separate renders. Only summaries of
the original sanitizer reports were retained; full raw sanitizer reports are
unavailable, and initcheck was not rerun today.

`historical-1070/` contains retained full-suite logs/failure lists. Historical
#1069/#1070 GPU suites retain the same 17 failures as main; they were not rerun
in full today. Those failures are 12 orthographic UT cases, four clamped
hit-distance cases, and missing PNG-compression dependencies. Original scripts
with absolute local paths are provenance records, not commands to run unchanged.
`historical-manifest.json` provides sizes and SHA-256 hashes of these records.
For #1068, the original four-file rendering suite retained 16 failures matching
main (`final_suites.out` and failure lists); that selected suite was not a full
repository run.

No new experiment is claimed for #1082 or #1083. Their previously published
loss/gradient and toy-optimization evidence remains applicable. In particular,
#1083 has no reproduction of #978's real-driving training setup.
