from typing import Sequence, Type

import torch

from magicsim.Env.Planner.Dwb import Dwb
from magicsim.Env.Planner.PController import PController
from magicsim.Env.Planner.Planner import Planner
from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Sensor.OccupancyManager import OccupancyManager


_DWB_TYPES = ("dwb_humanoid", "dwb_holonomic", "dwb_differential", "dwb_quadruped")


def _get_cfg_val(obj, name: str, default):
    """Read a field from OmegaConf / dict / attrs, matching Dwb/PController helpers."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return default


def _has_cfg_key(obj, name: str) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return name in obj
    if hasattr(obj, name):
        return True
    if hasattr(obj, "get"):
        try:
            return obj.get(name, None) is not None
        except Exception:
            return False
    return False


class Nav(Planner):
    """Router planner that combines PController and Dwb behind a single action channel.

    **Input width**: matches PController for the target robot (e.g. 15 for G1,
    8 for RidgebackFranka). The last column is the ``mode_flag``:

    * ``-2`` / ``-1`` / ``0`` / ``1`` — route to PController (same semantics as
      ``PController.step`` — skip / lock-skip / nav / turning).
    * ``2`` — route to Dwb. The first ``dwb_path_dim`` columns of the action
      are forwarded as path waypoints; remaining columns are ignored. The
      slice width is read from ``planner_cfg.base_action_space[<dwb_type>]``
      (e.g. 4 for ``dwb_humanoid``, 3 for mobile ``dwb_holonomic``) — never
      hardcoded per embodiment.

    **Output width** equals the PController output (``3 + n_extra_dims``) so
    the two branches can share the robot's expected input width (Dwb appends
    height / torso padding for humanoid/quadruped, matching PController's
    ``output_width``).
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        occupancy_manager: OccupancyManager | None,
        robot_type: str,
        robot_name: str,
        device: torch.device,
        num_envs: int,
        planner_cfg,
        robot_config,
        base_planner_config,
        p_controller_helper: Type | None = None,
        n_extra_dims: int | None = None,
    ):
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.device = device
        self.num_envs = num_envs

        # --- Resolve dwb variant (humanoid / holonomic / differential / quadruped). ---
        dwb_type = _get_cfg_val(base_planner_config, "dwb_type", None)
        if dwb_type is None:
            for k in _DWB_TYPES:
                if _has_cfg_key(base_planner_config, k):
                    dwb_type = k
                    break
        assert dwb_type is not None, (
            "Nav config must specify 'dwb_type' (e.g. 'dwb_humanoid', 'dwb_holonomic') "
            "or include a sub-dict named after the dwb variant."
        )
        dwb_type = dwb_type.lower()
        assert dwb_type in _DWB_TYPES, (
            f"Nav: unsupported dwb_type '{dwb_type}', expected one of {_DWB_TYPES}."
        )
        self.dwb_type = dwb_type

        # --- Dwb path slice width: read from planner_cfg, never hardcoded. ---
        dwb_space = planner_cfg.base_action_space[dwb_type]
        self.dwb_path_dim = int(dwb_space.shape[-1])

        # --- Nav input width: prefer explicit "nav" entry (e.g. G1PlannerCfg),
        #     otherwise fall back to PController's width. Either way, humanoid = 15,
        #     RidgebackFranka = 8 — the channel is shared across embodiments. ---
        _input_key = "nav" if "nav" in planner_cfg.base_action_dim else "p_controller"
        self.input_dim = int(planner_cfg.base_action_dim[_input_key])

        # --- Build inner PController (reuses the existing p_controller pipeline). ---
        p_controller_params = (
            _get_cfg_val(base_planner_config, "p_controller", {}) or {}
        )
        if p_controller_helper is None:
            p_controller_helper = getattr(planner_cfg, "p_controller_helper", None)
        if n_extra_dims is None:
            n_extra_dims = getattr(planner_cfg, "p_controller_n_extra_dims", 4)
        self.p_controller = PController(
            robot_manager=robot_manager,
            robot_type=robot_type,
            robot_name=robot_name,
            device=device,
            num_envs=num_envs,
            n_extra_dims=n_extra_dims,
            p_controller_helper=p_controller_helper,
            p_controller_config=p_controller_params,
        )

        # --- Build inner Dwb (reuses the existing dwb pipeline). ---
        # Device: prefer the per-dwb-config override (e.g. body.dwb_holonomic.device),
        # fall back to a top-level body.device, then to "cuda:0". Dwb itself also reads
        # ``device`` from its sub-config and overrides whatever we pass in here.
        dwb_sub_cfg = _get_cfg_val(base_planner_config, dwb_type, None)
        dwb_device_str = _get_cfg_val(
            dwb_sub_cfg, "device", _get_cfg_val(base_planner_config, "device", "cuda:0")
        )
        dwb_device = torch.device(str(dwb_device_str))
        self.dwb = Dwb(
            robot_manager=robot_manager,
            robot_name=robot_name,
            device=dwb_device,
            occupancy_manager=occupancy_manager,
            planner_cfg=planner_cfg,
            robot_config=robot_config,
            base_planner_type=dwb_type,
            base_planner_config=base_planner_config,
        )

        # Shared output width — PController and Dwb produce the same width per robot.
        self.output_width = int(self.p_controller.output_width)

    def _to_device_tensor(self, action) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        else:
            action = action.to(self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return action

    def step(
        self,
        action: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        """Route each env's action to PController or Dwb based on ``mode_flag``.

        ``action`` may be 2D ``[N, D]`` (single target frame, e.g. AutoCollect flow)
        or 3D ``[N, T, D]`` (path of waypoints, e.g. DWB test flow). The mode flag
        is taken from the last column (last waypoint when 3D).
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device)

        action = self._to_device_tensor(action)

        if action.ndim == 3:
            mode_flag = action[:, -1, -1]
        else:
            mode_flag = action[:, -1]

        dwb_mask = (mode_flag - 2.0).abs() < 0.5
        pc_mask = ~dwb_mask

        # if getattr(self.dwb, "debug", False):
        #     print(
        #         f"[Nav] action.shape={tuple(action.shape)} mode_flag={mode_flag.tolist()} "
        #         f"dwb_mask={dwb_mask.tolist()} pc_mask={pc_mask.tolist()}"
        #     )

        N = action.shape[0]
        out = torch.zeros(N, self.output_width, device=self.device)

        if torch.any(pc_mask):
            idx = torch.nonzero(pc_mask, as_tuple=False).squeeze(-1)
            pc_env_ids = env_ids[idx]
            pc_action = action[idx, -1, :] if action.ndim == 3 else action[idx, :]
            pc_out = self.p_controller.step(pc_action, pc_env_ids)
            out[idx] = pc_out.to(device=out.device, dtype=out.dtype)

        if torch.any(dwb_mask):
            idx = torch.nonzero(dwb_mask, as_tuple=False).squeeze(-1)
            # Dwb lives on cuda (expensive rollout); ship inputs to its device and
            # bring the output back to the sim device on assignment.
            dwb_env_ids = env_ids[idx].to(self.dwb.device)
            if action.ndim == 3:
                dwb_action = action[idx, :, : self.dwb_path_dim]
            else:
                dwb_action = action[idx, : self.dwb_path_dim].unsqueeze(1)
            dwb_action = dwb_action.to(self.dwb.device)
            dwb_out = self.dwb.step(dwb_action, dwb_env_ids)
            assert dwb_out.shape[-1] == self.output_width, (
                f"Nav: Dwb output width {dwb_out.shape[-1]} != PController output width "
                f"{self.output_width}. Cross-embodiment config mismatch."
            )
            out[idx] = dwb_out.to(device=out.device, dtype=out.dtype)

        return out

    def reset_idx(self, env_ids: Sequence[int]):
        self.p_controller.reset_idx(env_ids)
        self.dwb.reset_idx(env_ids)

    def update_obstacles(
        self,
        obstacle_avoidance_path_list: list = None,
        obstacle_ignore_path_list: list = None,
        env_ids: list = None,
    ):
        self.dwb.update_obstacles(
            obstacle_avoidance_path_list, obstacle_ignore_path_list, env_ids
        )
        self.p_controller.update_obstacles(
            obstacle_avoidance_path_list, obstacle_ignore_path_list, env_ids
        )
