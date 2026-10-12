import importlib.util
import os
import sys

import torch

binary = os.environ["GSPLAT_TEST_EXTENSION"]
spec = importlib.util.spec_from_file_location("gsplat_cuda", binary)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
sys.modules["gsplat.csrc"] = native
