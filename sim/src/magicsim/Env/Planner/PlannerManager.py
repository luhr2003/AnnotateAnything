from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple, Union
from omegaconf import DictConfig
from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Planner.PController import PController
from magicsim.Env.Planner.Planner import Planner
from magicsim.Env.Utils.file import Logger
import torch
import gymnasium as gym
import numpy as np
from prettytable import PrettyTable
from magicsim.Env.Sensor.OccupancyManager import OccupancyManager
from magicsim.Env.Planner.Services.MotionGenServer import MotionGenServer
from magicsim.Env.Planner.Services.IKServer import IKServer
from magicsim.Env.Planner.Services.DualMotionGenServer import DualMotionGenServer
from magicsim.Env.Planner.Services.DualIKServer import DualIKServer
from curobo.scene import Scene
from curobo.viewer import UsdWriter
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from magicsim.Env.Planner.Dwb import Dwb
from magicsim.Env.Planner.Nav import Nav

from curobo.config_io import load_yaml, join_path
from curobo.content import get_robot_configs_path


class PlannerManager:
    def __init__(
        self,
        num_envs: int,
        robot_config: DictConfig,
        device: torch.device,
        logger: Logger,
    ):
        """
        init corresponding parameters.
        """
        self.robot_manager = None
        self.occupancy_manager = None
        self.num_envs = num_envs
        self.robot_config = robot_config
        self.device = device
        self.logger = logger
        self.single_action_space = gym.spaces.Dict()
        self.planner_configs: Dict[str, Dict[str, DictConfig]] = {}
        self.total_action_dim = 0
        self.planner_dict: Dict[str, Dict[str, Planner]] = {}
        self.planner_type_dict: Dict[str, Dict[str, str]] = {}
        self.planner_slice_dict: Dict[str, Dict[str, Tuple[int, int]]] = {}
        # Output dim per slot (RobotManager input) - from robot_info; default equals input dim
        self.planner_output_dim_dict: Dict[str, Dict[str, int]] = {}
        # Single server per robot (post-MERGE_LEFT_RIGHT §1–§8 flatten).
        # ``hand_id`` in AtomicSkill / GlobalPlanner action headers no
        # longer selects a server — it only drives target packing
        # (NaN for inactive arm; see MERGE_LEFT_RIGHT.md §3).
        self.motiongen_server: Dict[
            str, Union[MotionGenServer, DualMotionGenServer]
        ] = {}
        self.ik_server: Dict[str, Union[IKServer, DualIKServer]] = {}
        # Cached world configs keyed by relative_to_world_frame flag.
        # Updated once per update_obstacles call, then shared across all servers.
        self.world_cfgs_cache: Dict[bool, List] = {}
        # Which relative_to_world_frame variants are needed (computed once at setup)
        self.needed_flags: set = set()
        # v2 replacement for v1's UsdHelper. ``UsdWriter`` exposes both the
        # obstacle parser (used by ``stage_obstacles_as_scene``) and the
        # ``add_world_to_stage`` debug visualizer.
        self.usd_helper: UsdWriter | None = None

    def initialize(
        self,
        robot_manager: RobotManager,
        occupancy_manager: OccupancyManager,
    ):
        """
        Initialize the planner after the world is initialized.

        Args:
            robot_manager: RobotManager instance
        """
        self.robot_manager = robot_manager
        self.occupancy_manager = occupancy_manager

    def step(
        self,
        action: torch.Tensor | dict[str, torch.Tensor] = None,
        env_ids: Sequence[int] = None,
    ):
        """
        Process all planners (robot) in a unified way

        Args:
            action: Robot action (optional)
            env_ids: List of environment IDs

        Returns:
            If only action is provided: returns planned robot action (backward compatible)
            If action is provided: returns planned robot action
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)
            else:
                env_ids = env_ids.to(self.device)

        action = self._flatten_actions(action, env_ids)
        processed_action = torch.tensor([], device=self.device)
        # print("action: ", action)
        # print("self.total_action_dim: ", self.total_action_dim)
        assert action.shape[0] == len(env_ids), (
            f"Action shape[0] {action.shape[0]} should be equal to env_ids length {len(env_ids)}"
        )
        if action.dim() == 2:
            assert action.shape[1] == self.total_action_dim, (
                f"Action shape[1] {action.shape[1]} should be equal to total_action_dim {self.total_action_dim}"
            )
        else:
            assert action.shape[2] == self.total_action_dim, (
                f"Action shape[2] {action.shape[2]} should be equal to total_action_dim {self.total_action_dim}"
            )
        planners = self._flatten_nested_dict_values(self.planner_dict)
        planner_slices = self._flatten_nested_dict_values(self.planner_slice_dict)
        planner_output_dims = self._flatten_nested_dict_values(
            self.planner_output_dim_dict
        )
        planner_keys = self._flatten_nested_dict_keys(self.planner_dict)
        for planner, (s, e), output_dim, (robot_name, slot) in zip(
            planners, planner_slices, planner_output_dims, planner_keys
        ):
            if planner is not None:
                if action.dim() == 2:
                    cur_processed_action = planner.step(action[:, s:e], env_ids)
                else:
                    cur_processed_action = planner.step(action[:, :, s:e], env_ids)
                assert cur_processed_action.shape[-1] == output_dim, (
                    f"[{robot_name}] {slot}: planner output dim "
                    f"{cur_processed_action.shape[-1]} != robot expected {output_dim}"
                )
                assert cur_processed_action.dim() == 2, (
                    f"Cur_processed action shape {cur_processed_action.shape} should be (N, {output_dim})"
                )
                # Ensure cur_processed_action is on the same device as processed_action
                cur_processed_action = cur_processed_action.to(self.device)
                processed_action = torch.cat(
                    [processed_action, cur_processed_action], dim=1
                )
            else:
                if action.dim() == 2:
                    action_slice = action[:, s:e]
                else:
                    # 3D (N, T, D): slice last dim, flatten to 2D to match planner output
                    action_slice = action[:, 0, s:e]
                if processed_action is None or processed_action.numel() == 0:
                    processed_action = action_slice
                else:
                    processed_action = torch.cat(
                        [processed_action, action_slice], dim=1
                    )
        return processed_action

    def reset_idx(self, env_ids):
        for robot_name, planners in self.planner_dict.items():
            for planner_name, planner in planners.items():
                if planner is not None:
                    planner.reset_idx(env_ids)

    def _flatten_actions(
        self,
        actions: torch.Tensor | dict[str, torch.Tensor],
        env_ids: torch.Tensor | Sequence[int],
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        batch_size = len(env_ids)
        if isinstance(actions, torch.Tensor):
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)
            if actions.shape[0] == self.num_envs and batch_size != self.num_envs:
                actions = actions[env_ids]
            return actions.to(self.device)
        chunks = []
        for rname, rspace in self.single_action_space.spaces.items():
            for k in rspace.spaces.keys():
                t = torch.as_tensor(
                    actions[rname][k], dtype=torch.float32, device=self.device
                )
                # Select env subset if tensor matches total env dimension
                if t.shape[0] == self.num_envs and batch_size != self.num_envs:
                    t = t[env_ids]
                # Ensure batch dimension exists
                if t.ndim == 1:
                    if t.shape[0] == batch_size:
                        t = t.unsqueeze(1)  # Treat as per-env scalar -> [N, 1]
                    elif t.shape[0] == self.num_envs:
                        t = t[env_ids].unsqueeze(1)
                    else:
                        t = t.unsqueeze(0)  # Fallback: treat as single sample
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)  # Flatten feature dimensions
                chunks.append(t)
        # Concatenate along feature dimension (dim=1), batch dimension (dim=0) must match
        return torch.cat(chunks, dim=1)

    def _flatten_nested_dict_values(self, data: Dict[str, Any]) -> List[Any]:
        flattened: List[Any] = []
        for value in data.values():
            if isinstance(value, dict):
                flattened.extend(self._flatten_nested_dict_values(value))
            else:
                flattened.append(value)
        return flattened

    def _flatten_nested_dict_keys(self, data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Returns [(robot_name, slot), ...] in the same order as _flatten_nested_dict_values."""
        result: List[Tuple[str, str]] = []
        for robot_name, robot_val in data.items():
            if isinstance(robot_val, dict):
                for slot in robot_val.keys():
                    result.append((robot_name, slot))
            else:
                result.append((robot_name, "?"))
        return result

    def sample_actions(self, batched: bool = True, env_ids: Sequence[int] = None):
        if env_ids is None:
            return (
                self.action_space.sample()
                if batched
                else self.single_action_space.sample()
            )
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim.device)
        sampled = {}
        for _ in range(len(env_ids)):
            one = self.single_action_space.sample()
            for rname, ract in one.items():
                if rname not in sampled:
                    sampled[rname] = {}
                for k, v in ract.items():
                    sampled[rname].setdefault(k, []).append(v)
        for rname in sampled:
            for k in sampled[rname]:
                sampled[rname][k] = torch.tensor(
                    np.array(sampled[rname][k]), dtype=torch.float32, device=self.device
                )
        return sampled

    def reset(self):
        self.setup_planner()
        env_idx = list(range(self.num_envs))
        for robot_name, planners in self.planner_dict.items():
            for planner_name, planner in planners.items():
                if planner is not None:
                    planner.reset_idx(env_idx)

    def setup_planner(self):
        offset = 0
        for robot_name, robot_config in self.robot_config.items():
            planner_config = robot_config.get("planner", None)
            self.planner_configs[robot_name] = planner_config
            self.planner_dict[robot_name] = {}
            self.planner_type_dict[robot_name] = {}
            self.planner_slice_dict[robot_name] = {}
            self.single_action_space[robot_name] = gym.spaces.Dict()
            cur_planner = {}
            robot_info = self.robot_manager.get_info()[robot_name]
            robot_cfg = self.robot_manager.robot_cfgs[robot_name]
            base_action_info = robot_info.get("base_action", None)
            arm_action_info = robot_info.get("arm_action", None)
            eef_action_info = robot_info.get("eef_action", None)
            # Output dim = RobotManager input dim (from robot_info); default equals input
            base_robot_dim = (
                base_action_info.get("action_dim", 0) if base_action_info else 0
            )
            arm_robot_dim = (
                arm_action_info.get("action_dim", 0) if arm_action_info else 0
            )
            eef_robot_dim = (
                eef_action_info.get("action_dim", 0) if eef_action_info else 0
            )
            self.planner_output_dim_dict[robot_name] = {
                "base": base_robot_dim,
                "arm": arm_robot_dim,
                "eef": eef_robot_dim,
            }
            if base_action_info is not None:
                base_action_space = base_action_info.get("action_space", None)
                base_action_dim = base_action_info.get("action_dim", 0)
            else:
                base_action_space = None
                base_action_dim = 0
            if arm_action_info is not None:
                arm_action_space = arm_action_info.get("action_space", None)
                arm_action_dim = arm_action_info.get("action_dim", 0)
            else:
                arm_action_space = None
                arm_action_dim = 0
            if eef_action_info is not None:
                eef_action_space = eef_action_info.get("action_space", None)
                eef_action_dim = eef_action_info.get("action_dim", 0)
            else:
                eef_action_space = None
                eef_action_dim = 0
            if planner_config is None:
                cur_planner = {"base": None, "arm": None, "eef": None}
                self.planner_dict[robot_name] = cur_planner
                self.planner_type_dict[robot_name] = {
                    "base": None,
                    "arm": None,
                    "eef": None,
                }
                self.planner_slice_dict[robot_name] = {
                    "base": (offset, offset + base_action_dim),
                    "arm": (
                        offset + base_action_dim,
                        offset + base_action_dim + arm_action_dim,
                    ),
                    "eef": (
                        offset + base_action_dim + arm_action_dim,
                        offset + base_action_dim + arm_action_dim + eef_action_dim,
                    ),
                }
                offset += base_action_dim + arm_action_dim + eef_action_dim
                if base_action_space is not None:
                    self.single_action_space[robot_name]["base_action"] = (
                        base_action_space
                    )
                if arm_action_space is not None:
                    self.single_action_space[robot_name]["arm_action"] = (
                        arm_action_space
                    )
                if eef_action_space is not None:
                    self.single_action_space[robot_name]["eef_action"] = (
                        eef_action_space
                    )
            else:
                # Check for dual mode configuration
                dual_mode = planner_config.get("dual_mode", False)
                base_joint_names = planner_config.get("base_joint_names", [])

                if planner_config.get("ik", None) is not None:
                    ik_config = planner_config.get("ik", None)
                    if ik_config.get("type", None) is None:
                        raise ValueError(
                            f"Robot '{robot_name}' planner.ik missing `type:` "
                            f"(got keys {list(ik_config.keys())})."
                        )
                    if (
                        ik_config.get("enable", False)
                        and robot_name not in self.ik_server
                    ):
                        srv = self._create_ik_server(
                            robot_name,
                            ik_config,
                            dual_mode,
                            base_joint_names,
                        )
                        srv.eef_num = int(ik_config.get("eef_num", 1))
                        self.ik_server[robot_name] = srv

                if planner_config.get("motiongen", None) is not None:
                    motiongen_config = planner_config.get("motiongen", None)
                    if motiongen_config.get("type", None) is None:
                        raise ValueError(
                            f"Robot '{robot_name}' planner.motiongen missing "
                            f"`type:` (got keys {list(motiongen_config.keys())})."
                        )
                    if (
                        motiongen_config.get("enable", False)
                        and robot_name not in self.motiongen_server
                    ):
                        srv = self._create_motiongen_server(
                            robot_name,
                            motiongen_config,
                            dual_mode,
                            base_joint_names,
                        )
                        srv.eef_num = int(motiongen_config.get("eef_num", 1))
                        self.motiongen_server[robot_name] = srv
                base_planner_config = planner_config.get("base", None)
                if base_planner_config is None:
                    base_planner_config = planner_config.get("body", None)
                if base_planner_config is not None:
                    base_planner_type = base_planner_config.get("type", None)
                    if (
                        base_planner_type is None
                        or base_planner_type.lower() == "default"
                    ):
                        cur_planner["base"] = None
                        self.planner_dict[robot_name]["base"] = cur_planner["base"]
                        self.planner_type_dict[robot_name]["base"] = None
                        self.planner_slice_dict[robot_name]["base"] = (
                            offset,
                            offset + base_action_dim,
                        )
                        offset += base_action_dim
                        if base_action_space is not None:
                            self.single_action_space[robot_name]["base_action"] = (
                                base_action_space
                            )
                    elif base_planner_type.lower() == "p_controller":
                        # Get P-controller parameters from config
                        p_controller_params = base_planner_config.get(
                            "p_controller", {}
                        )
                        assert len(p_controller_params) > 0, (
                            "P-controller parameters are not set in the config"
                        )
                        print(
                            f"[PlannerManager] P-controller params from config: {dict(p_controller_params) if p_controller_params else {}}"
                        )
                        # Get p_controller_helper class from planner config
                        cur_planner_cfg = robot_cfg.planner
                        p_controller_helper = getattr(
                            cur_planner_cfg, "p_controller_helper", None
                        )
                        n_extra_dims = getattr(
                            cur_planner_cfg, "p_controller_n_extra_dims", 4
                        )
                        cur_planner["base"] = PController(
                            robot_manager=self.robot_manager,
                            robot_type=robot_config.name,
                            robot_name=robot_name,
                            device=self.device,
                            num_envs=self.num_envs,
                            n_extra_dims=n_extra_dims,
                            p_controller_helper=p_controller_helper,
                            p_controller_config=p_controller_params,
                        )
                        # P-controller input is robot-specific (e.g., 15D for G1, 8D for RidgebackFranka)
                        # Preprocessed by preprocess_fn to [x, y, heading, *extra, mode_flag]
                        # Output is (3 + n_extra_dims)D: [vel_x, vel_y, ang_vel, *extra]
                        base_action_dim = cur_planner_cfg.base_action_dim[
                            base_planner_type.lower()
                        ]
                        base_action_space = cur_planner_cfg.base_action_space[
                            base_planner_type.lower()
                        ]
                        self.planner_dict[robot_name]["base"] = cur_planner["base"]
                        self.planner_type_dict[robot_name]["base"] = base_planner_type
                        self.planner_slice_dict[robot_name]["base"] = (
                            offset,
                            offset + base_action_dim,
                        )
                        offset += base_action_dim
                        self.single_action_space[robot_name]["base_action"] = (
                            gym.spaces.Box(
                                low=base_action_space[0].cpu().numpy(),
                                high=base_action_space[1].cpu().numpy(),
                                shape=(base_action_dim,),
                            )
                        )
                    elif base_planner_type.lower() in ("dwb_holonomic", "dwb_humanoid"):
                        cur_planner_cfg = robot_cfg.planner
                        # Dwb device: read body.<dwb_type>.device, fall back to body.device,
                        # then default cuda:0. Dwb's own ctor will also re-read it.
                        _dwb_sub = base_planner_config.get(
                            base_planner_type.lower(), None
                        ) or getattr(
                            base_planner_config, base_planner_type.lower(), None
                        )
                        _device_str = (
                            (
                                _dwb_sub.get("device", None)
                                if isinstance(_dwb_sub, dict)
                                else getattr(_dwb_sub, "device", None)
                            )
                            if _dwb_sub is not None
                            else None
                        )
                        if _device_str is None:
                            _device_str = (
                                base_planner_config.get("device", None)
                                if isinstance(base_planner_config, dict)
                                else getattr(base_planner_config, "device", None)
                            )
                        if _device_str is None:
                            _device_str = "cuda:0"
                        cur_planner["base"] = Dwb(
                            robot_manager=self.robot_manager,
                            robot_name=robot_name,
                            device=torch.device(str(_device_str)),
                            occupancy_manager=self.occupancy_manager,
                            planner_cfg=cur_planner_cfg,
                            robot_config=robot_config,
                            base_planner_type=base_planner_type,
                            base_planner_config=base_planner_config,
                        )
                        key = base_planner_type.lower()
                        base_action_dim = cur_planner_cfg.base_action_dim[key]
                        base_action_space = cur_planner_cfg.base_action_space[key]
                        self.planner_dict[robot_name]["base"] = cur_planner["base"]
                        self.planner_type_dict[robot_name]["base"] = base_planner_type
                        self.planner_slice_dict[robot_name]["base"] = (
                            offset,
                            offset + base_action_dim,
                        )
                        offset += base_action_dim
                        self.single_action_space[robot_name]["base_action"] = (
                            gym.spaces.Box(
                                low=base_action_space[0].cpu().numpy(),
                                high=base_action_space[1].cpu().numpy(),
                                shape=(base_action_dim,),
                            )
                        )
                    elif base_planner_type.lower() == "nav":
                        cur_planner_cfg = robot_cfg.planner
                        p_controller_helper = getattr(
                            cur_planner_cfg, "p_controller_helper", None
                        )
                        n_extra_dims = getattr(
                            cur_planner_cfg, "p_controller_n_extra_dims", 4
                        )
                        cur_planner["base"] = Nav(
                            robot_manager=self.robot_manager,
                            occupancy_manager=self.occupancy_manager,
                            robot_type=robot_config.name,
                            robot_name=robot_name,
                            device=self.device,
                            num_envs=self.num_envs,
                            planner_cfg=cur_planner_cfg,
                            robot_config=robot_config,
                            base_planner_config=base_planner_config,
                            p_controller_helper=p_controller_helper,
                            n_extra_dims=n_extra_dims,
                        )
                        # Nav shares its input width with PController (e.g. 15 for G1,
                        # 8 for RidgebackFranka). Mode flag in the last column decides
                        # whether the row is routed to PController ({-2,-1,0,1}) or
                        # Dwb (2). The Dwb slice width is read from the dwb sub-config.
                        # Prefer an explicit "nav" entry in the per-robot planner cfg
                        # (e.g. G1PlannerCfg); otherwise fall back to "p_controller".
                        if "nav" in cur_planner_cfg.base_action_dim:
                            key = "nav"
                        else:
                            key = "p_controller"
                        base_action_dim = cur_planner_cfg.base_action_dim[key]
                        base_action_space = cur_planner_cfg.base_action_space[key]
                        self.planner_dict[robot_name]["base"] = cur_planner["base"]
                        self.planner_type_dict[robot_name]["base"] = base_planner_type
                        self.planner_slice_dict[robot_name]["base"] = (
                            offset,
                            offset + base_action_dim,
                        )
                        offset += base_action_dim
                        self.single_action_space[robot_name]["base_action"] = (
                            gym.spaces.Box(
                                low=base_action_space[0].cpu().numpy(),
                                high=base_action_space[1].cpu().numpy(),
                                shape=(base_action_dim,),
                            )
                        )
                    else:
                        raise NotImplementedError(
                            f"Planner type {base_planner_type} not supported."
                        )
                else:
                    cur_planner["base"] = None
                    base_action_dim = 0
                    self.planner_dict[robot_name]["base"] = cur_planner["base"]
                    self.planner_type_dict[robot_name]["base"] = None
                    self.planner_slice_dict[robot_name]["base"] = (
                        offset,
                        offset + base_action_dim,
                    )
                    offset += base_action_dim

                arm_planner_config = planner_config.get("arm", None)
                if arm_planner_config is not None:
                    arm_planner_type = arm_planner_config.get("type", None)
                    if (
                        arm_planner_type is None
                        or arm_planner_type.lower() == "default"
                    ):
                        cur_planner["arm"] = None
                        self.planner_dict[robot_name]["arm"] = cur_planner["arm"]
                        self.planner_type_dict[robot_name]["arm"] = None
                        self.planner_slice_dict[robot_name]["arm"] = (
                            offset,
                            offset + arm_action_dim,
                        )
                        offset += arm_action_dim
                        self.single_action_space[robot_name]["arm_action"] = (
                            arm_action_space
                        )
                    elif arm_planner_type.lower() == "curobo":
                        # v1 had a ``Curobo`` wrapper class
                        # (``magicsim.Env.Planner.Curobo``) that exposed a
                        # ``Planner``-compatible API on top of a
                        # MotionGenServer. The user deleted that module
                        # in the v2 migration; re-wiring the arm-planner
                        # slot directly against MotionGenServer is a
                        # follow-up (``ServiceMigrate.md`` TODO).
                        raise NotImplementedError(
                            "arm_planner_type='curobo' is not wired in the v2 "
                            "migration. The legacy Curobo wrapper "
                            "(``magicsim.Env.Planner.Curobo``) has been removed; "
                            "re-wire this branch against MotionGenServer directly "
                            "(see ServiceMigrate.md) before enabling it."
                        )
                        cur_planner_cfg = robot_cfg.planner
                        arm_action_space = cur_planner_cfg.arm_action_space[
                            arm_planner_type.lower()
                        ]
                        arm_action_dim = cur_planner_cfg.arm_action_dim[
                            arm_planner_type.lower()
                        ]
                        self.planner_dict[robot_name]["arm"] = cur_planner["arm"]
                        self.planner_type_dict[robot_name]["arm"] = arm_planner_type
                        self.planner_slice_dict[robot_name]["arm"] = (
                            offset,
                            offset + arm_action_dim,
                        )
                        offset += arm_action_dim
                        self.single_action_space[robot_name]["arm_action"] = (
                            gym.spaces.Box(
                                low=arm_action_space[0].cpu().numpy(),
                                high=arm_action_space[1].cpu().numpy(),
                                shape=(arm_action_dim,),
                            )
                        )
                    else:
                        raise NotImplementedError(
                            f"Planner type {arm_planner_type} not supported."
                        )

                eef_planner_config = planner_config.get("eef", None)
                if eef_planner_config is not None:
                    eef_planner_type = eef_planner_config.get("type", None)
                    if (
                        eef_planner_type is None
                        or eef_planner_type.lower() == "default"
                    ):
                        cur_planner["eef"] = None
                        self.planner_dict[robot_name]["eef"] = cur_planner["eef"]
                        self.planner_type_dict[robot_name]["eef"] = None
                        self.planner_slice_dict[robot_name]["eef"] = (
                            offset,
                            offset + eef_action_dim,
                        )
                        offset += eef_action_dim
                        self.single_action_space[robot_name]["eef_action"] = (
                            eef_action_space
                        )
                    else:
                        raise NotImplementedError(
                            f"Planner type {eef_planner_type} not supported."
                        )
        self.total_action_dim = offset
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        print(self._get_summary_table())

        # Pre-compute which relative_to_world_frame flags are needed for obstacle updates
        self.needed_flags = set()
        for srv in self.motiongen_server.values():
            if getattr(srv, "dual_mode", False):
                self.needed_flags.update({True, False})
            else:
                self.needed_flags.add(srv.relative_to_world_frame)
        for srv in self.ik_server.values():
            if getattr(srv, "dual_mode", False):
                self.needed_flags.update({True, False})
            else:
                self.needed_flags.add(srv.relative_to_world_frame)

    def _get_summary_table(self) -> str:
        msg = f"pm:  <PlannerManager> total_action_dim={self.total_action_dim}\n"
        table = PrettyTable()
        table.title = f"Planner Slots (total input: {self.total_action_dim})"
        table.field_names = [
            "Robot",
            "Slot",
            "Planner",
            "Input Dim",
            "Input Slice",
            "Output Dim",
        ]
        table.align["Robot"] = "l"
        table.align["Slot"] = "l"
        table.align["Planner"] = "l"
        table.align["Input Dim"] = "r"
        table.align["Input Slice"] = "r"
        table.align["Output Dim"] = "r"
        for robot_name in self.planner_dict:
            for slot in self.planner_dict[robot_name]:
                planner_type = self.planner_type_dict.get(robot_name, {}).get(
                    slot, None
                )
                s, e = self.planner_slice_dict.get(robot_name, {}).get(slot, (0, 0))
                output_dim = self.planner_output_dim_dict.get(robot_name, {}).get(
                    slot, 0
                )
                table.add_row(
                    [
                        robot_name,
                        slot,
                        planner_type or "passthrough",
                        e - s,
                        f"[{s}:{e}]",
                        output_dim,
                    ]
                )
        msg += table.get_string() + "\n"
        return msg

    @staticmethod
    def _merge_extra_fk_link(robot_yaml: dict) -> Tuple[dict, List[str], List[str]]:
        """Read ``extra_fk_link`` + ``info_links`` from YAML top level, merge
        the former into ``robot_cfg.kinematics.tool_frames``, and return the
        mutated robot_cfg together with both lists for downstream forwarding.

        Contract (see ``extra_info_links.md``):
          - YAML ``robot_cfg.kinematics.tool_frames`` = TRACKED frames only
            (this is what callers pass targets for; ``len(tool_frames)`` is
            the caller-visible tracked count).
          - YAML top-level ``extra_fk_link`` = FK-only frames (optional). We
            merge them into ``tool_frames`` so cuRobo allocates FK buffers;
            the Server applies ``ToolPoseCriteria.disabled()`` to each so
            they contribute ZERO cost at IK seeding, main IK optimizer, and
            MotionGen trajopt.
          - YAML top-level ``info_links`` = position-mode readout order
            (optional). If omitted, default = original tracked tool_frames
            order.

        The mutation is in-place on a deepcopied ``robot_cfg`` so different
        robots don't cross-contaminate if PlannerManager reloads the YAML.
        """
        robot_cfg = deepcopy(robot_yaml["robot_cfg"])
        extra_fk_link = list(robot_yaml.get("extra_fk_link", []) or [])
        info_links = robot_yaml.get("info_links", None)

        kin = robot_cfg.setdefault("kinematics", {})
        tracked = list(kin.get("tool_frames", []) or [])
        # Dedup while preserving order: tracked first, then any extras not
        # already declared in tracked.
        merged = list(tracked)
        for f in extra_fk_link:
            if f not in merged:
                merged.append(f)
        kin["tool_frames"] = merged
        return robot_cfg, extra_fk_link, info_links

    def _create_motiongen_server(
        self,
        robot_name: str,
        motiongen_config: DictConfig,
        dual_mode: bool = False,
        base_joint_names: List[str] = None,
    ) -> Union[MotionGenServer, DualMotionGenServer]:
        """Create a MotionGenServer or DualMotionGenServer for the given robot_name.

        Args:
            robot_name: Name of the robot
            motiongen_config: MotionGen configuration
            dual_mode: If True, create DualMotionGenServer
            base_joint_names: Virtual base joint names for dual mode

        Returns:
            MotionGenServer or DualMotionGenServer instance
        """
        assert motiongen_config is not None, "MotionGen config is not set in the config"
        assert motiongen_config.get("type", None) is not None, (
            "MotionGen type is not set in the config"
        )
        robot_type = motiongen_config.get("type", None)

        # Get num_instances from config if available, otherwise use default
        num_instances = motiongen_config.get(
            "motiongen_num_instances", 1
        )  # Default 1 instances
        microbatch_wait_ms = motiongen_config.get("microbatch_wait_ms", 200.0)
        batch_size = motiongen_config.get("batch_size", 8)
        debug = motiongen_config.get("debug", False)
        mode = motiongen_config.get("mode", "joint")
        relative_to_world_frame = motiongen_config.get("relative_to_world_frame", True)

        if dual_mode:
            # For dual mode: load both locked and free configs
            robot_yml_file_locked = f"magicsim_{robot_type.lower()}.yml"
            robot_yml_file_free = f"magicsim_{robot_type.lower()}_mobile.yml"

            robot_yaml_locked = load_yaml(
                join_path(get_robot_configs_path(), robot_yml_file_locked)
            )
            robot_yaml_free = load_yaml(
                join_path(get_robot_configs_path(), robot_yml_file_free)
            )

            # Locked + free YAMLs are the SAME robot (only the base config
            # differs), so ``extra_fk_link`` / ``info_links`` / tracked
            # ``tool_frames`` must match. Raise if they don't — silent
            # union would hide a config bug.
            robot_cfg_locked, locked_extra, locked_info = self._merge_extra_fk_link(
                robot_yaml_locked,
            )
            robot_cfg_free, extra_fk_link, info_links = self._merge_extra_fk_link(
                robot_yaml_free,
            )
            if locked_extra != extra_fk_link:
                raise ValueError(
                    f"{robot_yml_file_locked!r} extra_fk_link={locked_extra} "
                    f"!= {robot_yml_file_free!r} extra_fk_link={extra_fk_link}. "
                    f"Locked + free YAMLs of the same robot must declare "
                    f"identical extra_fk_link."
                )
            if locked_info != info_links:
                raise ValueError(
                    f"{robot_yml_file_locked!r} info_links={locked_info} "
                    f"!= {robot_yml_file_free!r} info_links={info_links}. "
                    f"Locked + free YAMLs of the same robot must declare "
                    f"identical info_links."
                )
            if (
                robot_cfg_locked["kinematics"]["tool_frames"]
                != robot_cfg_free["kinematics"]["tool_frames"]
            ):
                raise ValueError(
                    f"{robot_yml_file_locked!r} kinematics.tool_frames="
                    f"{robot_cfg_locked['kinematics']['tool_frames']} "
                    f"!= {robot_yml_file_free!r} kinematics.tool_frames="
                    f"{robot_cfg_free['kinematics']['tool_frames']}. "
                    f"Locked + free YAMLs must declare identical tool_frames."
                )

            # Use FREE config for robot parameters (has add_joints for base)
            robot_lock_joints = robot_cfg_free["kinematics"].get("lock_joints", None)
            robot_ignore_joints = robot_yaml_free.get("ignore_joints", {})
            robot_add_joints = robot_yaml_free.get("add_joints", {})

            robot_dof_name = list(self.robot_manager.robots[robot_name].joint_names)
            if robot_add_joints is not None:
                robot_dof_name.extend(list(robot_add_joints.keys()))
            if robot_lock_joints is not None:
                robot_dof_name_active = [
                    x for x in robot_dof_name if x not in robot_lock_joints
                ]
            else:
                robot_dof_name_active = robot_dof_name
            if robot_ignore_joints is not None:
                robot_dof_name_active = [
                    x for x in robot_dof_name_active if x not in robot_ignore_joints
                ]

            # Per-arm joint groups + ``pin_inactive_arm`` flag — see
            # DualIKServer / DualMotionGenServer for the contract. Read
            # off the FREE yaml; defaults to no pinning when the
            # robot's yaml doesn't declare these fields.
            mg_left_arm_joints = list(robot_yaml_free.get("left_arm_joints", []) or [])
            mg_right_arm_joints = list(
                robot_yaml_free.get("right_arm_joints", []) or []
            )
            mg_pin_inactive_arm = bool(robot_yaml_free.get("pin_inactive_arm", False))

            return DualMotionGenServer(
                robot_manager=self.robot_manager,
                robot_cfg_locked=robot_cfg_locked,
                robot_cfg_free=robot_cfg_free,
                robot_name=robot_name,
                robot_dof_name=robot_dof_name,
                robot_dof_name_active=robot_dof_name_active,
                robot_lock_joints=robot_lock_joints,
                robot_ignore_joints=robot_ignore_joints,
                robot_add_joints=robot_add_joints,
                device=self.device,
                base_joint_names=base_joint_names or [],
                batch_size=batch_size,
                microbatch_wait_ms=microbatch_wait_ms,
                num_instances=num_instances,
                debug=debug,
                mode=mode,
                info_links=info_links,
                extra_fk_link=extra_fk_link,
                left_arm_joints=mg_left_arm_joints,
                right_arm_joints=mg_right_arm_joints,
                pin_inactive_arm=mg_pin_inactive_arm,
            )
        else:
            # Standard single mode
            robot_yml_file = f"magicsim_{robot_type.lower()}.yml"
            robot_yaml = load_yaml(join_path(get_robot_configs_path(), robot_yml_file))
            robot_cfg, extra_fk_link, info_links = self._merge_extra_fk_link(robot_yaml)
            robot_lock_joints = robot_cfg["kinematics"].get("lock_joints", None)
            robot_ignore_joints = robot_yaml.get("ignore_joints", {})
            robot_add_joints = robot_yaml.get("add_joints", {})

            robot_dof_name = list(self.robot_manager.robots[robot_name].joint_names)
            if robot_add_joints is not None:
                robot_dof_name.extend(list(robot_add_joints.keys()))
            if robot_lock_joints is not None:
                robot_dof_name_active = [
                    x for x in robot_dof_name if x not in robot_lock_joints
                ]
            else:
                robot_dof_name_active = robot_dof_name
            if robot_ignore_joints is not None:
                robot_dof_name_active = [
                    x for x in robot_dof_name_active if x not in robot_ignore_joints
                ]

            return MotionGenServer(
                robot_manager=self.robot_manager,
                robot_cfg=robot_cfg,
                robot_name=robot_name,
                robot_dof_name=robot_dof_name,
                robot_dof_name_active=robot_dof_name_active,
                robot_lock_joints=robot_lock_joints,
                robot_ignore_joints=robot_ignore_joints,
                device=self.device,
                batch_size=batch_size,
                microbatch_wait_ms=microbatch_wait_ms,
                num_instances=num_instances,
                debug=debug,
                mode=mode,
                info_links=info_links,
                extra_fk_link=extra_fk_link,
                robot_add_joints=robot_add_joints,
                relative_to_world_frame=relative_to_world_frame,
            )

    def _create_ik_server(
        self,
        robot_name: str,
        ik_config: DictConfig,
        dual_mode: bool = False,
        base_joint_names: List[str] = None,
    ) -> Union[IKServer, DualIKServer]:
        """Create an IKServer or DualIKServer for the given robot_name.

        Args:
            robot_name: Name of the robot
            ik_config: IK configuration
            dual_mode: If True, create DualIKServer
            base_joint_names: Virtual base joint names for dual mode

        Returns:
            IKServer or DualIKServer instance
        """
        assert ik_config is not None, "IK config is not set in the config"
        assert ik_config.get("type", None) is not None, (
            "IK type is not set in the config"
        )
        robot_type = ik_config.get("type", None)

        # Get IK server parameters from config if available, otherwise use defaults
        num_instances = ik_config.get("ik_num_instances", 1)
        batch_size = ik_config.get("ik_batch_size", 8)
        num_seeds = ik_config.get("ik_num_seeds", 20)
        position_threshold = ik_config.get("ik_position_threshold", 0.005)
        rotation_threshold = ik_config.get("ik_rotation_threshold", 0.05)
        microbatch_wait_ms = ik_config.get("microbatch_wait_ms", 200.0)
        relative_to_world_frame = ik_config.get("relative_to_world_frame", True)
        debug = ik_config.get("debug", False)
        # Largest goalset G the IK Server is allowed to handle. cuRobo
        # asserts ``num_goalset <= max_goalset`` at solve time. Default 10000
        # accommodates DexGrasp / Grasp candidate sweeps; lower it per-robot
        # via ``ik_max_goalset`` to shave warmup cost (warmup runs at G).
        max_goalset = int(ik_config.get("ik_max_goalset", 10000))
        # ``paired`` defaults TRUE — see MERGE_LEFT_RIGHT.md §9. Single-frame
        # robots no-op (paired with L=1 reduces to unpaired argmin); dual-arm
        # robots pick up paired argmin out of the box. Set ``paired: false``
        # in the YAML only for robots that explicitly want independent picks.
        paired = bool(ik_config.get("paired", True))

        if dual_mode:
            # For dual mode: load both locked and free configs (same as DualMotionGenServer)
            robot_yml_file_locked = f"magicsim_{robot_type.lower()}.yml"
            robot_yml_file_free = f"magicsim_{robot_type.lower()}_mobile.yml"

            robot_yaml_locked = load_yaml(
                join_path(get_robot_configs_path(), robot_yml_file_locked)
            )
            robot_yaml_free = load_yaml(
                join_path(get_robot_configs_path(), robot_yml_file_free)
            )

            # Locked + free YAMLs are the SAME robot — see
            # ``_create_motiongen_server`` for the rationale. Raise on
            # misalignment instead of silently unioning.
            robot_cfg_locked, locked_extra, locked_info = self._merge_extra_fk_link(
                robot_yaml_locked,
            )
            robot_cfg_free, extra_fk_link, info_links = self._merge_extra_fk_link(
                robot_yaml_free,
            )
            if locked_extra != extra_fk_link:
                raise ValueError(
                    f"{robot_yml_file_locked!r} extra_fk_link={locked_extra} "
                    f"!= {robot_yml_file_free!r} extra_fk_link={extra_fk_link}. "
                    f"Locked + free YAMLs of the same robot must declare "
                    f"identical extra_fk_link."
                )
            if locked_info != info_links:
                raise ValueError(
                    f"{robot_yml_file_locked!r} info_links={locked_info} "
                    f"!= {robot_yml_file_free!r} info_links={info_links}. "
                    f"Locked + free YAMLs of the same robot must declare "
                    f"identical info_links."
                )
            if (
                robot_cfg_locked["kinematics"]["tool_frames"]
                != robot_cfg_free["kinematics"]["tool_frames"]
            ):
                raise ValueError(
                    f"{robot_yml_file_locked!r} kinematics.tool_frames="
                    f"{robot_cfg_locked['kinematics']['tool_frames']} "
                    f"!= {robot_yml_file_free!r} kinematics.tool_frames="
                    f"{robot_cfg_free['kinematics']['tool_frames']}. "
                    f"Locked + free YAMLs must declare identical tool_frames."
                )

            robot_lock_joints = robot_cfg_free["kinematics"].get("lock_joints", None)
            robot_ignore_joints = robot_yaml_free.get("ignore_joints", {})
            robot_add_joints = robot_yaml_free.get("add_joints", {})

            robot_dof_name = list(self.robot_manager.robots[robot_name].joint_names)
            if robot_add_joints is not None:
                robot_dof_name.extend(list(robot_add_joints.keys()))
            if robot_lock_joints is not None:
                robot_dof_name_active = [
                    x for x in robot_dof_name if x not in robot_lock_joints
                ]
            else:
                robot_dof_name_active = robot_dof_name
            if robot_ignore_joints is not None:
                robot_dof_name_active = [
                    x for x in robot_dof_name_active if x not in robot_ignore_joints
                ]

            # Per-arm joint groups + ``pin_inactive_arm`` flag (vega
            # ``magicsim_vega1p_sharpa{,_mobile}.yml`` declares these so
            # the Server can pin a tool's arm joints at sim's live
            # values when the caller disables that tool slot via NaN.
            # Robots without these fields default to no pinning.
            left_arm_joints = list(robot_yaml_free.get("left_arm_joints", []) or [])
            right_arm_joints = list(robot_yaml_free.get("right_arm_joints", []) or [])
            pin_inactive_arm = bool(robot_yaml_free.get("pin_inactive_arm", False))

            return DualIKServer(
                robot_manager=self.robot_manager,
                robot_cfg_locked=robot_cfg_locked,
                robot_cfg_free=robot_cfg_free,
                robot_name=robot_name,
                device=self.device,
                base_joint_names=base_joint_names or [],
                batch_size=batch_size,
                microbatch_wait_ms=microbatch_wait_ms,
                num_instances=num_instances,
                # DualIKServer takes per-side seed counts (locked uses fewer
                # because its DOF is lower; free has virtual base joints).
                # Use ``num_seeds`` for locked and bump free to ≥50 per v1.
                num_seeds_locked=num_seeds,
                num_seeds_free=max(int(num_seeds), 50),
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                max_goalset=max_goalset,
                robot_add_joints=robot_add_joints,
                robot_ignore_joints=robot_ignore_joints,
                robot_lock_joints=robot_lock_joints,
                robot_dof_name_active=robot_dof_name_active,
                extra_fk_link=extra_fk_link,
                info_links=info_links,
                left_arm_joints=left_arm_joints,
                right_arm_joints=right_arm_joints,
                pin_inactive_arm=pin_inactive_arm,
                debug=debug,
                paired=paired,
            )
        else:
            # Standard single mode
            robot_yml_file = f"magicsim_{robot_type.lower()}.yml"
            robot_yaml = load_yaml(join_path(get_robot_configs_path(), robot_yml_file))
            robot_cfg, extra_fk_link, info_links = self._merge_extra_fk_link(robot_yaml)

            return IKServer(
                robot_manager=self.robot_manager,
                robot_cfg=robot_cfg,
                robot_name=robot_type,
                device=self.device,
                batch_size=batch_size,
                microbatch_wait_ms=microbatch_wait_ms,
                num_instances=num_instances,
                num_seeds=num_seeds,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                max_goalset=max_goalset,
                extra_fk_link=extra_fk_link,
                info_links=info_links,
                relative_to_world_frame=relative_to_world_frame,
                paired=paired,
                debug=debug,
            )

    def update_obstacles(
        self,
        obstacle_avoidance_path_list: list = None,
        obstacle_ignore_path_list: list = None,
        env_ids: list = None,
        motiongen_only: bool = False,
    ):
        """
        Args:
            obstacle_avoidance_path_list: list of obstacle avoidance paths
            obstacle_ignore_path_list: list of obstacle ignore paths
            env_ids: list of env ids to update obstacles
            motiongen_only: if True, only update MotionGen world (skip IK and planners)
        """
        if env_ids is None:
            env_ids = range(self.num_envs)
        env_ids = list(env_ids)

        # ---- 1. Build / refresh the shared world_cfgs cache ----
        self.world_cfgs_cache.clear()
        for flag in self.needed_flags:
            self.world_cfgs_cache[flag] = self.get_world_cfgs(
                obstacle_avoidance_path_list,
                obstacle_ignore_path_list,
                env_ids,
                relative_to_world_frame=flag,
            )

        # ---- 2. Distribute cached world_cfgs to MotionGen servers ----
        # One server per robot (see MERGE_LEFT_RIGHT.md §1–§8).
        for motiongen in self.motiongen_server.values():
            if getattr(motiongen, "dual_mode", False):
                motiongen.update_world_dual(
                    env_ids,
                    self.world_cfgs_cache[True],
                    self.world_cfgs_cache[False],
                )
            else:
                motiongen.update_world(
                    env_ids,
                    self.world_cfgs_cache[motiongen.relative_to_world_frame],
                )

        if motiongen_only:
            return

        # ---- 3. Distribute to IK servers ----
        for ik_srv in self.ik_server.values():
            if getattr(ik_srv, "dual_mode", False):
                ik_srv.update_world_dual(
                    env_ids,
                    self.world_cfgs_cache[True],
                    self.world_cfgs_cache[False],
                )
            else:
                ik_srv.update_world(
                    env_ids,
                    self.world_cfgs_cache[ik_srv.relative_to_world_frame],
                )

        # ---- 4. Update planners ----
        # v1 had ``isinstance(planner, Curobo)`` here to skip the Curobo
        # wrapper (its obstacle update happened via the server). The
        # Curobo wrapper was deleted; any arm planner in ``planner_dict``
        # today exposes ``update_obstacles`` directly, so the isinstance
        # skip goes away.
        for _, planners in self.planner_dict.items():
            for planner in planners.values():
                if planner is None:
                    continue
                update = getattr(planner, "update_obstacles", None)
                if update is None:
                    continue
                update(obstacle_avoidance_path_list, obstacle_ignore_path_list, env_ids)

    def get_world_cfgs(
        self,
        obstacle_avoidance_path_list: list = None,
        obstacle_ignore_path_list: list = None,
        env_ids: list = None,
        relative_to_world_frame: bool = True,
    ) -> List[Scene]:
        """Get scene configs for collision checking (one per env).

        v2 notes: returns :class:`curobo.scene.Scene` instances instead of
        v1 ``WorldConfig``. The Services accept either Scene or SceneCfg
        dict in ``update_world``; Scene is the direct upstream of
        ``scene_collision_checker.load_collision_model``.

        Reference frame per branch (see MERGE_LEFT_RIGHT.md §9 / Services
        README §3.2 for the full kinematics↔collision frame contract):

        +---------------------------+--------------------------------+-----------------------------------+
        | ``relative_to_world_frame``| Reference prim                 | Consumer                          |
        +---------------------------+--------------------------------+-----------------------------------+
        | ``True``                  | ``/World/envs/env_i/Robot_0``  | Locked-base solver. Targets       |
        |                           | (robot-base frame)             | transformed world→robot in        |
        |                           |                                | ``_preprocess_request``; collision|
        |                           |                                | must match → robot frame.         |
        +---------------------------+--------------------------------+-----------------------------------+
        | ``False``                 | ``/World/envs/env_i``          | Free-base solver (dual servers,   |
        |                           | (sub-env frame)                | mobile manipulators). Virtual     |
        |                           |                                | base joints ``dummy_base_*`` are  |
        |                           |                                | sub-env relative → zero joint     |
        |                           |                                | values place the robot at the     |
        |                           |                                | sub-env origin. Collision obs     |
        |                           |                                | must ALSO be sub-env-relative     |
        |                           |                                | (NOT raw world frame, or the      |
        |                           |                                | collision check sees obstacles    |
        |                           |                                | displaced by the env's world      |
        |                           |                                | offset — common bug in            |
        |                           |                                | multi-env scenes).                |
        +---------------------------+--------------------------------+-----------------------------------+

        Args:
            obstacle_avoidance_path_list: List of obstacle paths to avoid.
            obstacle_ignore_path_list: List of obstacle paths to ignore.
            env_ids: List of environment IDs.
            relative_to_world_frame: If True → robot-base frame. If False →
                sub-env frame (NOT world frame — see table above).

        Returns:
            List of :class:`Scene` (one per env_id).
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        stage = get_current_stage()
        if self.usd_helper is None:
            self.usd_helper = UsdWriter()
            self.usd_helper.load_stage(stage)
        world_cfgs: List[Scene] = []
        for i in env_ids:
            if obstacle_avoidance_path_list is None:
                world_cfgs.append(Scene())
                continue
            obstacle_avoidance_paths = [
                f"/World/envs/env_{i}/{path}" for path in obstacle_avoidance_path_list
            ]
            obstacle_ignore_paths = [f"/World/envs/env_{i}/Robot_0"]
            if obstacle_ignore_path_list is not None:
                obstacle_ignore_paths.extend(obstacle_ignore_path_list)
            # Reference frame — always sub-env or robot, NEVER raw world
            # (would displace obstacles by env offset in multi-env scenes).
            reference = (
                f"/World/envs/env_{i}/Robot_0"
                if relative_to_world_frame
                else f"/World/envs/env_{i}"
            )
            # v2 equivalent of v1's
            # ``UsdHelper.get_obstacles_from_stage(...).get_collision_check_world()``.
            # Inline the 2-line ``stage_obstacles_as_scene`` helper from
            # ``curobo.examples.isaacsim.helper`` — importing that module on
            # Isaac Sim 5.1 fails because the helper module pulls in the
            # deprecated ``omni.isaac.core`` namespace, but the function body
            # itself only needs the public ``UsdWriter`` API.
            scene = self.usd_helper.get_obstacles_from_stage(
                only_paths=list(obstacle_avoidance_paths),
                ignore_substring=list(obstacle_ignore_paths)
                if obstacle_ignore_paths
                else None,
                reference_prim_path=reference,
            )
            # CRITICAL: convert spheres / cylinders / capsules → meshes. v2's
            # ``scene_collision_checker.load_collision_model`` (→
            # ``DataScene.load_from_scene_cfg``) only iterates over ``cuboid``,
            # ``mesh``, and ``voxel``; every other obstacle type is SILENTLY
            # DROPPED. v1 did the same conversion via its own
            # ``get_collision_check_world()``; we missed porting that call.
            scene = scene.get_collision_check_world()
            world_cfgs.append(scene)
        return world_cfgs

    def debug_visualize_world_cfgs(
        self,
        env_id: int = 0,
        relative_to_world_frame: bool = True,
        obstacles_frame: str = "debug_collision",
    ):
        """Debug: 把 world_cfg 画到 Isaac Sim stage 上，用于查看碰撞几何。

        Args:
            env_id: 要可视化的环境 ID
            relative_to_world_frame: 使用 world_cfgs_cache 中哪个坐标系
            obstacles_frame: 在 base_frame 下的 prim 名称
        """
        if not self.world_cfgs_cache:
            return
        flag = relative_to_world_frame
        if flag not in self.world_cfgs_cache:
            return
        world_cfg = self.world_cfgs_cache[flag][env_id]
        if self.usd_helper is None:
            self.usd_helper = UsdWriter()
            self.usd_helper.load_stage(get_current_stage())
        base_frame = f"/World/envs/env_{env_id}"
        debug_prim_path = f"{base_frame}/{obstacles_frame}"
        if is_prim_path_valid(debug_prim_path):
            delete_prim(debug_prim_path)
        # v2 equivalent of v1 ``UsdHelper.add_world_to_stage`` — Scene in,
        # same kwarg shape.
        self.usd_helper.add_world_to_stage(
            world_cfg,
            base_frame=base_frame,
            obstacles_frame=obstacles_frame,
        )

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for robot_name in self.planner_dict.keys():
            info[robot_name] = {}
            for part in ("base", "arm", "eef"):
                planner_type = self.planner_type_dict.get(robot_name, {}).get(part)
                action_slice = self.planner_slice_dict.get(robot_name, {}).get(part)
                action_dim = 0
                if action_slice is not None:
                    action_dim = int(action_slice[1] - action_slice[0])
                info[robot_name][part] = {
                    "planner_type": planner_type,
                    "action_slice": action_slice,
                    "action_dim": action_dim,
                }
        info["robot_name_list"] = list(self.planner_dict.keys())
        return info

    def get_move_strategy(self, robot_name: str):
        """Get the move_strategy function and distance_threshold for a robot.

        Returns:
            Tuple of (move_strategy_fn, distance_threshold) or (None, 0.3)
            move_strategy_fn signature: (trajectory: Tensor, robot_state: Dict) -> Tensor
        """
        robot_cfg = self.robot_manager.robot_cfgs.get(robot_name, None)
        if robot_cfg is not None and hasattr(robot_cfg, "planner"):
            planner_cfg = robot_cfg.planner
            move_strategy = getattr(planner_cfg, "move_strategy", None)
            distance_threshold = getattr(
                planner_cfg, "move_strategy_distance_threshold", 0.3
            )
            return move_strategy, distance_threshold
        return None, 0.3

    def get_dehatch_strategy(self, robot_name: str):
        """Get the dehatch_strategy function for a robot.

        Returns:
            dehatch_strategy callable or None.
            Signature: (robot_state: Dict, **kwargs) -> Tensor
        """
        robot_cfg = self.robot_manager.robot_cfgs.get(robot_name, None)
        if robot_cfg is not None and hasattr(robot_cfg, "planner"):
            return getattr(robot_cfg.planner, "dehatch_strategy", None)
        return None
