"""HighTorque Mini-Pi getup task built on mjlab."""

import warp as wp

# libmathdx (cuBLASDx/cuSolverDx) only supports sm_70+. mujoco_warp's tile_cholesky and
# tile_matmul fail to compile on older GPUs (e.g. GTX 10xx), so fall back to Warp's
# native tile implementations there.
if any(d.arch < 70 for d in wp.get_cuda_devices()):
  wp.config.enable_mathdx_solver = False
  wp.config.enable_mathdx_gemm = False

from minipi_getup.ftsr.config import *  # noqa: E402, F401, F403
from minipi_getup.getup.config.minipi import *  # noqa: E402, F401, F403
import minipi_getup.ftsr_ref.config  # noqa: E402, F401
