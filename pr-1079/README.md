# PR #1079 rendering evidence

Measured CUDA rendering comparisons for [nerfstudio-project/gsplat#1079](https://github.com/nerfstudio-project/gsplat/pull/1079).

- Before: `512d366b67073d77ca099ede742683c165dfc23b`.
- After: `065f4a8105e45a7de9117ce561f7a0f5b14c6c22`.
- Hardware: NVIDIA B200; driver 580.126.20. Measurements use one GPU, float32, PyTorch 2.11.0+cu128, CUDA 12.8.
- Configurations: 3DGS and 2DGS, each packed/unpacked. No batch dimensions in these figures; the PR's regression suite separately covers batch dimensions, multiple color layouts and render modes, and distributed rendering.

## Measurements and figures

`results/metrics.json` and `results/metrics.csv` contain measured maxima for all four configurations. Per-run JSON records output metadata shapes, device/dtype, finite checks, and gradient checks. The `.npz` files contain original RGB, alpha, depth and nonempty input gradients. `.status.json` records each subprocess's exit code and signal.

- `empty-scene.png`: N=0, two cameras, 128×96, SH degree 0, RGB+ED. Camera backgrounds are `[0.10, 0.65, 0.75]` and `[0.85, 0.45, 0.10]`. Before terminates with SIGFPE and produces **no rendered frame**; its panel is an annotation. After returns those backgrounds, zero alpha/depth, correctly shaped empty gradients, and background gradients of 12,288 per channel for a summed RGB loss. 2DGS normal, depth-derived normal, distortion and median outputs are also checked for exact zero.
- `control-rgb.png` and `control-depth.png`: N=1008, one camera, 192×144. The deterministic synthetic wavy surface uses 28×36 Gaussians, identity quaternions, scales `[0.041, 0.041, 0.025]`, opacity 0.8, SH degree 0 colors, and no supplied background. RGB and expected depth are rendered using RGB+ED. Before/after panels share the same display scales; error panels show measured absolute differences with a 1e-6 minimum display range.
- Control gradients are computed separately from `mean(RGB * linspace(0.5, 1.5, H*W).reshape(1,H,W,1))`, with distortion loss disabled. The metrics compare means, quaternions, scales, opacity and SH color gradients. The table's gradient delta is the largest absolute difference over these tensors; per-tensor differences are in the JSON.

These controls use existing supported nonempty layouts. SH degree 0 avoids independent direct-color layout defects; RGB-only backward avoids the existing 2DGS depth-gradient contiguity issue. They are numerical regression evidence, not a scene-quality benchmark or a claim about every nonempty configuration.

## Reproduce

Requires CUDA, a compatible PyTorch installation, NumPy and Matplotlib. Run the same script against separately built source checkouts. Each render executes in a subprocess with a 180-second timeout and core dumps disabled. The collector verifies SIGFPE for the before empty scene and normal exit for every other run.

```bash
# Save this assets checkout's path before moving between source checkouts.
ASSET_DIR=/absolute/path/to/pr_asset/pr-1079
RESULTS_DIR=/absolute/path/to/new-results

# In the before checkout at 512d366:
export BUILD_2DGS=1 BUILD_3DGS=1 BUILD_3DGUT=1 BUILD_CAMERA_WRAPPERS=1
export NUM_CHANNELS=1,3,4 TORCH_CUDA_ARCH_LIST=10.0 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export TORCH_EXTENSIONS_DIR=/tmp/gsplat-pr1079-before
python "$ASSET_DIR/render_comparison.py" --version before --output "$RESULTS_DIR"

# In the after checkout at 065f4a8 (same build flags):
export PYTHONPATH="$PWD"
export TORCH_EXTENSIONS_DIR=/tmp/gsplat-pr1079-after
python "$ASSET_DIR/render_comparison.py" --version after --output "$RESULTS_DIR"

# Compare the two runs and regenerate figures:
python "$ASSET_DIR/render_comparison.py" --summarize --output "$RESULTS_DIR"
```

Set the CUDA architecture for your GPU when reproducing on other hardware. Alternatively, pass `--extension /path/to/gsplat_cuda.so` to load a prebuilt extension without JIT compilation. The measurements here used this option. The before extension rebuilt all five changed native translation units from 512d366 and reused unchanged object files with identical headers and compiler flags. The after extension was the verified full build from 065f4a8. Python rendering code differs only in documentation between these commits. `provenance.json` records source commits and binary SHA-256 checksums; binary files are not published.
