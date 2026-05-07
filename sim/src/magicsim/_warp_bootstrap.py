"""Pin pip ``warp-lang`` (>= 1.12) into ``sys.modules`` BEFORE Isaac Sim.

Why
----
Isaac Sim 5.1 ships ``omni.warp.core 1.8.2`` via ``isaacsim/extscache/`` and
prepends that directory onto ``sys.path`` when the kit boots. The older
Warp does not support the ``wp.func(..., module=...)`` kwarg that cuRobo
2.0 uses for overload registration in ``curobo._src.geom.collision``
kernels, so every Isaac Sim launch of a cuRobo call crashes with
``TypeError: func() got an unexpected keyword argument 'module'``.

Importing ``warp`` (and every submodule cuRobo touches) before
``SimulationApp(...)`` caches the pip ``warp-lang`` in ``sys.modules``.
Subsequent ``import warp`` statements — including cuRobo's and anything
Isaac Sim itself pulls in via ``omni.warp.core`` — short-circuit to that
cached module.

Mirrors ``curobo.examples.isaacsim.bootstrap``; see
``CUROBO_V2_04_ENV_MIGRATION.md`` §4 for the full rationale.

Usage
-----
Import this module **before** any ``SimulationApp``/``AppLauncher``
instantiation. The MagicLauncher does this at the top of its module.
For scripts that bypass MagicLauncher, add ``import magicsim._warp_bootstrap``
above the ``SimulationApp(...)`` line.
"""

import warnings as _warnings

# Suppress repetitive Warp deprecation warnings from downstream callers we
# don't control (cuRobo internals and Isaac Sim's warp.torch shim access
# ``warp.torch.device_from_torch`` / ``warp.types.warp_type_to_np_dtype``,
# which moved to ``warp.device_from_torch`` / ``warp._src.types`` in
# warp-lang ≥ 1.12). These aren't actionable on our side — filtering
# here silences the logspam without hiding warnings we emit ourselves.
_warnings.filterwarnings(
    "ignore", message=r".*warp\.torch\..*", category=DeprecationWarning
)
_warnings.filterwarnings(
    "ignore", message=r".*warp\.types\..*", category=DeprecationWarning
)

# Side-effect imports — cache pip warp-lang in sys.modules before Isaac Sim
# can insert its extscache onto sys.path or mutate ``warp.__path__``.
# Every submodule cuRobo touches must be pinned, not just the top-level.
import warp  # noqa: F401, E402
import warp.torch  # noqa: F401, E402
import warp.context  # noqa: F401, E402
import warp.config  # noqa: F401, E402

# Populate ``warp.context.runtime`` now. Some downstream accessors
# (e.g. ``wp.torch.device_from_torch``) read ``warp.context.runtime``
# before cuRobo's own ``init_warp()`` has had a chance to run.
warp.init()
