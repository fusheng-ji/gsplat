"""Run the fixed Lidar regression against an old gsplat Python checkout."""

import argparse
import importlib.util
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="Old gsplat checkout root")
parser.add_argument("--test-file", required=True, help="Fixed test_rasterization.py")
args = parser.parse_args()
sys.path.insert(0, args.source)

import gsplat

print("Python source:", gsplat.__file__, flush=True)
spec = importlib.util.spec_from_file_location("lidar_validation", args.test_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for model in ("at128", "pandar128"):
    try:
        module.test_rasterization_lidar_tiling_fits_tile_size(model)
    except AssertionError:
        print(model, "EXPECTED OLD RENDERING FAILURE", flush=True)
        traceback.print_exc()
    else:
        raise RuntimeError(f"{model}: old implementation unexpectedly passed")
