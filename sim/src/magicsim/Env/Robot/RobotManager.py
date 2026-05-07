from typing import List, Sequence
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import numpy as np
import torch
import re

from magicsim.Env.Utils.file import Logger
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from magicsim.Env.Robot.Cfg import ROBOT_DICT
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationManager
from magicsim.Env.Robot.mdp.action_manager import ActionManager
from magicsim.Env.Sensor.frame_transformer import FrameTransformer
from magicsim.Env.Utils.rotations import euler_angles_to_quat
from isaaclab.utils.noise.noise_model import NoiseModel
from isaaclab.utils.noise.noise_cfg import NoiseModelCfg
from magicsim.Env.Robot.Cfg.Base import NOISE_TYPE_DICT
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import get_current_stage
from magicsim.Env.Environment.Utils.Basic import seed_everywhere
from isaaclab.sensors import SensorBase
from isaaclab.sensors.camera import TiledCamera

try:
    from magicsim.Env.Sensor.Tactile.TactileManager import TactileManager
except ImportError:
    print("TactileManager could not be imported. Make sure tacex is installed.")


class RobotManager:
    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        device,
        logger: Logger,
        seeds_per_env: Sequence[int] | None = None,
    ):
        self.sim: IsaacRLEnv = None
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.logger = logger
        self.robot_cfgs = {}
        self.robot_types = {}
        self.robots: dict[str, Articulation] = {}
        self.robot_id: dict[str, int] = {}
        self.ee_frames: list[FrameTransformer] = []
        self.action_managers: dict[str, ActionManager] = {}
        self.observation_managers: dict[str, ObservationManager] = {}
        self.single_action_space = gym.spaces.Dict()
        self.action_space: gym.spaces.Dict = None
        self.single_observation_space = None
        self.observation_space = None
        self._action_noise_models: list[NoiseModel | None] = []
        self._obs_noise_models: list[NoiseModel | None] = []
        self._seeds_per_env: list[int] | None = None
        self.update_env_seeds(seeds_per_env)
        self.sensors: List[dict[str, SensorBase]] = []

        # Initialize tactile manager
        if self.config.get("Tactile", None) is not None:
            self.tactile_manager = TactileManager(
                num_envs=self.num_envs,
                config=self.config.Tactile,
                device=self.device,
                robot_manager=self,
            )
        else:
            self.tactile_manager = None

        OmegaConf.set_struct(self.config, False)
        self.config.pop("Tactile", None)

    def set_noise_model(self, noise_cfg: DictConfig):
        if "action" in noise_cfg:
            t = str(noise_cfg.action.type)
            nm = NOISE_TYPE_DICT[t](
                mean=noise_cfg.action.params.mean, std=noise_cfg.action.params.std
            )
            nm.operation = noise_cfg.action.operation
            am = NoiseModelCfg(noise_cfg=nm)
            action_noise_model = NoiseModel(
                noise_model_cfg=am, num_envs=self.num_envs, device=str(self.device)
            )
        else:
            action_noise_model = None
        if "observation" in noise_cfg:
            t = str(noise_cfg.observation.type)
            nm = NOISE_TYPE_DICT[t](
                mean=noise_cfg.observation.params.mean,
                std=noise_cfg.observation.params.std,
            )
            nm.operation = noise_cfg.observation.operation
            om = NoiseModelCfg(noise_cfg=nm)
            obs_noise_model = NoiseModel(
                noise_model_cfg=om, num_envs=self.num_envs, device=str(self.device)
            )
        else:
            obs_noise_model = None
        return action_noise_model, obs_noise_model

    def update_env_seeds(self, seeds: Sequence[int] | None):
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(
                f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}."
            )
        self._seeds_per_env = seed_list

    def _set_env_seed(self, env_id: int | None = None):
        if self._seeds_per_env is None:
            return
        if env_id is None:
            env_id = 0
        if env_id < 0 or env_id >= len(self._seeds_per_env):
            raise IndexError(
                f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seeds_per_env)})."
            )
        seed_everywhere(self._seeds_per_env[env_id])

    def _map_robot_name_to_robot_config(
        self,
        robot_name: str,
        robot_type: str,
        prim_path: str,
        asset_name: str,
        ori: torch.Tensor,
        pos: torch.Tensor,
        joint_pos: dict[str, float] | None,
        robot_config: DictConfig,
        sim: IsaacRLEnv,
    ):
        sensor_dict = {}
        cls = ROBOT_DICT.get(robot_name.lower())
        if cls is None:
            raise ValueError(f"Robot {robot_name} not supported.")
        if robot_type == "manipulator":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                frame_name=f"{asset_name}_ee",
                arm_action_name=robot_config.action.arm,
                eef_action_name=robot_config.action.eef,
            )
            ee_frame = FrameTransformer(cfg.ee_frame)
            self.ee_frames.append(ee_frame)
            sim.scene.sensors[f"{asset_name}_ee"] = ee_frame
        elif robot_type == "dualmanipulator":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                frame_name=f"{asset_name}_ee",
                arm_action_name=robot_config.action.arm,
                eef_action_name=robot_config.action.eef,
            )
            ee_frame = FrameTransformer(cfg.ee_frame)
            self.ee_frames.append(ee_frame)
            sim.scene.sensors[f"{asset_name}_ee"] = ee_frame
        elif robot_type == "dexterous":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                frame_name=f"{asset_name}_ee",
                base_action_name=getattr(robot_config.action, "base_action", None),
                arm_action_name=getattr(robot_config.action, "arm_action", None),
                eef_action_name=getattr(robot_config.action, "eef_action", None),
            )
            ee_frame = FrameTransformer(cfg.ee_frame)
            self.ee_frames.append(ee_frame)
            sim.scene.sensors[f"{asset_name}_ee"] = ee_frame
        elif robot_type == "mobile":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                base_action_name=robot_config.action.base,
            )
        elif robot_type == "mobilemanip":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                frame_name=f"{asset_name}_ee",
                base_action_name=robot_config.action.base_action,
                arm_action_name=robot_config.action.arm_action,
                eef_action_name=robot_config.action.eef_action,
            )
            ee_frame = FrameTransformer(cfg.ee_frame)
            self.ee_frames.append(ee_frame)
            sim.scene.sensors[f"{asset_name}_ee"] = ee_frame
        elif robot_type == "humanoid":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                base_action_name=robot_config.action.base_action,
                arm_action_name=robot_config.action.arm_action,
                eef_action_name=robot_config.action.eef_action,
            )
            sensor_dict[f"{asset_name}_head_camera"] = (TiledCamera, cfg.sensor)
        elif robot_type == "quadruped":
            cfg = cls(
                prim_path=prim_path,
                asset_name=asset_name,
                base_action_name=robot_config.action.base_action,
            )
        q = euler_angles_to_quat(ori, degrees=True)
        if pos is not None:
            cfg.robot.init_state.pos = pos
        if ori is not None:
            cfg.robot.init_state.rot = q
        if joint_pos is not None:
            cfg.robot.init_state.joint_pos = dict(joint_pos)
        return cfg, sensor_dict

    def reset(self):
        self.post_init()
        for m in self._action_noise_models:
            if m is not None:
                m.reset()
        for m in self._obs_noise_models:
            if m is not None:
                m.reset()

    def initialize(self, sim: IsaacRLEnv):
        cur_rob_id = 0
        if self._seeds_per_env:
            self._set_env_seed(0)
        for i, (robot_name_str, args) in enumerate(self.config.items()):
            robot_name = args["name"]
            robot_type = args["type"]
            asset_name = f"Robot_{i}"
            prim_path = f"{sim.scene.env_regex_ns}/{asset_name}"
            init_pos = args.common.get("initial_pos_range", [0, 0, 0, 0, 0, 0])
            init_pos = torch.from_numpy(
                np.random.uniform(low=init_pos[:3], high=init_pos[3:6])
            ).to(self.device)
            init_ori = args.common.get("initial_ori_range", [0, 0, 0, 0, 0, 0])
            init_ori = torch.from_numpy(
                np.random.uniform(low=init_ori[:3], high=init_ori[3:6])
            ).to(self.device)

            robot_cfg, sensor_dict = self._map_robot_name_to_robot_config(
                robot_name,
                robot_type,
                prim_path=prim_path,
                asset_name=asset_name,
                ori=init_ori,
                pos=init_pos,
                joint_pos=args["common"].get("initial_joint_pos", None),
                robot_config=args,
                sim=sim,
            )
            if args.get("noise", None) is not None:
                a, b = self.set_noise_model(args.noise)
                self._action_noise_models.append(a)
                self._obs_noise_models.append(b)
            else:
                self._action_noise_models.append(None)
                self._obs_noise_models.append(None)
            self.robot_cfgs[robot_name_str] = robot_cfg
            self.robot_types[robot_name_str] = robot_cfg.type
            robot = Articulation(robot_cfg.robot)

            self.robots[robot_name_str] = robot
            self.robot_id[robot_name_str] = cur_rob_id
            cur_rob_id += 1
            sim.scene.articulations[asset_name] = robot

            semantic_label = self._get_robot_semantic_label(
                robot_name_str, args, args["type"]
            )
            if semantic_label:
                self._add_semantic_label(robot_cfg, semantic_label)
        for sensor_name, (sensor_class, sensor_cfg) in sensor_dict.items():
            sensor = sensor_class(cfg=sensor_cfg)
            sim.scene.sensors[sensor_name] = sensor
            self.sensors.append(sensor)
        self.sim = sim
        if self.tactile_manager is not None:
            self.tactile_manager.initialize(sim)

    def reset_robot(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.sim.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(self.sim.device, dtype=torch.int32)
        else:
            env_ids_tensor = torch.tensor(
                [int(e) for e in env_ids], device=self.sim.device, dtype=torch.int32
            )
        env_id_list = env_ids_tensor.detach().cpu().tolist()

        for robot_name, robot in self.robots.items():
            n = len(env_id_list)
            cfg = list(self.config.values())[self.robot_id[robot_name]]

            joint_pos = robot.data.default_joint_pos[env_ids_tensor]
            joint_vel = torch.zeros_like(joint_pos, device=self.sim.device)

            robot.set_joint_position_target(joint_pos, env_ids=env_ids_tensor)
            robot.set_joint_velocity_target(joint_vel, env_ids=env_ids_tensor)
            robot.write_joint_state_to_sim(
                position=joint_pos, velocity=joint_vel, env_ids=env_ids_tensor
            )
            root_vel = torch.zeros((n, 6), dtype=torch.float32, device=self.device)
            robot.write_root_com_velocity_to_sim(
                root_velocity=root_vel, env_ids=env_ids_tensor
            )
            pos_range_cfg = cfg.common.get("initial_pos_range", [0, 0, 0, 0, 0, 0])
            ori_range_cfg = cfg.common.get("initial_ori_range", [0, 0, 0, 0, 0, 0])
            pr = np.array(pos_range_cfg, np.float32).reshape(-1, 6)
            orr = np.array(ori_range_cfg, np.float32).reshape(-1, 6)
            pr = np.tile(pr, (n, 1))
            orr = np.tile(orr, (n, 1))
            pos_samples = []
            ori_samples = []
            for idx, env_id in enumerate(env_id_list):
                self._set_env_seed(env_id)
                pos_samples.append(
                    np.random.uniform(low=pr[idx, :3], high=pr[idx, 3:6])
                )
                ori_samples.append(
                    np.random.uniform(low=orr[idx, :3], high=orr[idx, 3:6])
                )
            if pos_samples:
                pos_array = np.stack(pos_samples, axis=0)
                ori_array = np.stack(ori_samples, axis=0)
            else:
                pos_array = np.empty((0, 3), dtype=np.float32)
                ori_array = np.empty((0, 3), dtype=np.float32)
            reset_pos = (
                torch.from_numpy(pos_array).to(torch.float32).to(self.device)
                + self.sim.scene.env_origins[env_ids_tensor]
            )
            reset_ori = torch.from_numpy(ori_array).to(torch.float32).to(self.device)
            reset_quat = euler_angles_to_quat(reset_ori, degrees=True).to(self.device)
            robot.write_root_pose_to_sim(
                root_pose=torch.concatenate([reset_pos, reset_quat], axis=-1),
                env_ids=env_ids_tensor,
            )
            # update default root state for reset consistency
            default_root_state = torch.zeros(
                (n, 13), dtype=torch.float32, device=self.device
            )
            default_root_state[:, :3] = reset_pos
            default_root_state[:, 3:7] = reset_quat
            default_root_state[:, 7:] = root_vel
            robot._data.default_root_state[env_ids_tensor] = default_root_state

    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_id_list = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_id_list = env_ids.detach().cpu().tolist()
        else:
            env_id_list = [int(e) for e in env_ids]
        env_ids_tensor = torch.tensor(
            env_id_list, device=self.sim.device, dtype=torch.int32
        )
        self.reset_robot(env_ids_tensor)
        for om in self.observation_managers.values():
            om.reset(env_ids=env_id_list)
        for am in self.action_managers.values():
            am.reset(env_ids=env_id_list)
        for m in self._action_noise_models:
            if m is not None:
                m.reset(env_ids=env_ids_tensor)
        for m in self._obs_noise_models:
            if m is not None:
                m.reset(env_ids=env_ids_tensor)

    def post_init(self):
        self._robot_slices = []
        self.single_observation_space = gym.spaces.Dict()
        offset = 0
        for robot_name, robot_cfg in self.robot_cfgs.items():
            am = ActionManager(robot_cfg.action, self.sim)
            print("am: ", am)
            dim = int(am.total_action_dim)
            self.action_managers[robot_name] = am
            self.single_action_space[robot_name] = am.action_space
            self._robot_slices.append((offset, offset + dim))
            if hasattr(robot_cfg, "obs") and hasattr(
                robot_cfg.obs, "enable_corruption"
            ):
                robot_cfg.obs.enable_corruption = True
            om = ObservationManager({robot_name: robot_cfg.obs}, self.sim)
            self.observation_managers[robot_name] = om
            for group_name, term_names in om.active_terms.items():
                has_cat = om.group_obs_concatenate[group_name]
                group_dim = om.group_obs_dim[group_name]
                if has_cat:
                    self.single_observation_space[group_name] = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=group_dim
                    )
                else:
                    self.single_observation_space[group_name] = gym.spaces.Dict(
                        {
                            f"{t}_{robot_name}": gym.spaces.Box(
                                low=-np.inf, high=np.inf, shape=d
                            )
                            for t, d in zip(term_names, group_dim)
                        }
                    )
            offset += dim

        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.total_action_dim = offset
        self.reset_robot()

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

    def step(
        self,
        action: torch.Tensor | dict[str, torch.Tensor],
        env_ids: Sequence[int] = None,
        noise_flag: bool = True,
    ):
        if action is None:
            self.sim.sim_step()
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.sim.device)
            else:
                env_ids = env_ids.to(self.device)
        if action is not None:
            action = self._flatten_actions(action)
            assert action.shape[1] == self.total_action_dim, (
                f"Expected action shape (N, {self.total_action_dim}), got {action.shape}"
            )
            assert action.shape[0] == len(env_ids), (
                f"Expected action shape (N, {self.total_action_dim}), got {action.shape}"
            )
            action = self._flatten_actions(action)
            if noise_flag:
                action = self._apply_action_noise(action)
            self.sim.step(action, env_ids)

        # Note here we return the raw and processed action for env_ids
        return {
            "raw_action": self.get_raw_action(env_ids=env_ids),
            "processed_action": self.get_processed_action(env_ids=env_ids),
        }

    def get_raw_action(self, env_ids: Sequence[int] = None):
        """Get raw actions for each robot from their action managers.

        Args:
            env_ids: The environment ids to get actions for. If None, returns actions for all environments.

        Returns:
            A dictionary mapping robot names to their raw actions dictionaries.
            Each robot's dictionary contains term names as keys and raw action tensors as values.
        """
        raw_actions_dict = {}

        # Iterate through robots and their corresponding action managers
        for robot_name, action_manager in zip(
            self.robot_cfgs.keys(), self.action_managers.values()
        ):
            raw_actions_dict[robot_name] = action_manager.get_raw_action(
                env_ids=env_ids
            )

        return raw_actions_dict

    def get_processed_action(self, env_ids: Sequence[int] = None):
        """Get processed actions for each robot from their action managers.

        Args:
            env_ids: The environment ids to get actions for. If None, returns actions for all environments.

        Returns:
            A dictionary mapping robot names to their processed actions dictionaries.
            Each robot's dictionary contains term names as keys and processed action tensors as values.
        """
        processed_actions_dict = {}

        # Iterate through robots and their corresponding action managers
        for robot_name, action_manager in zip(
            self.robot_cfgs.keys(), self.action_managers.values()
        ):
            processed_actions_dict[robot_name] = action_manager.get_processed_action(
                env_ids=env_ids
            )

        return processed_actions_dict

    def get_robot_state(self, noise_flag: bool = False):
        obs_list = [om.compute() for om in self.observation_managers.values()]
        if noise_flag:
            obs_list = self._apply_obs_noise(obs_list)
        return obs_list

    def pre_physics_step(
        self,
        sim: IsaacRLEnv,
        actions: torch.Tensor | dict[str, torch.Tensor],
        env_ids: Sequence[int] = None,
    ):
        if actions is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim.device)
        actions = self._flatten_actions(actions)
        assert actions.shape[1] == self.total_action_dim
        for am, (s, e) in zip(self.action_managers.values(), self._robot_slices):
            am.process_action(actions[:, s:e], env_ids)

    def _apply_action(self, sim: IsaacRLEnv):
        for am in self.action_managers.values():
            am.apply_action()

    def _flatten_actions(self, actions: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(actions, torch.Tensor):
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)
            return actions.to(self.device)
        chunks = []
        for rname, rspace in self.single_action_space.spaces.items():
            for k in rspace.spaces.keys():
                t = torch.as_tensor(
                    actions[rname][k], dtype=torch.float32, device=self.device
                )
                if t.ndim == 1:
                    t = t.unsqueeze(1)
                if t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                chunks.append(t)
        return torch.cat(chunks, dim=1)

    def _apply_obs_noise(self, obs_per_robot: list):
        def map_tensors(x, f):
            if torch.is_tensor(x):
                return f(x)
            if isinstance(x, dict):
                return {k: map_tensors(v, f) for k, v in x.items()}
            return x

        for i, m in enumerate(self._obs_noise_models):
            if m is None:
                continue
            obs_per_robot[i] = map_tensors(obs_per_robot[i], lambda t: m.apply(t))
        return obs_per_robot

    def _apply_action_noise(self, actions_flat: torch.Tensor):
        chunks = []
        for (s, e), m in zip(self._robot_slices, self._action_noise_models):
            c = actions_flat[:, s:e]
            if m is not None:
                c = m.apply(c)
            chunks.append(c)
        return torch.cat(chunks, dim=1)

    # _add_semantic_labels_to_robots removed: labeling handled during initialize per env_0 root prim

    def _get_robot_semantic_label(
        self, robot_name: str, robot_config: DictConfig, robot_type: str
    ) -> str:
        """Get semantic label from config or infer from robot type."""
        # Check if semantic_label is explicitly set in config
        if hasattr(robot_config, "semantic_label") and robot_config.semantic_label:
            return str(robot_config.semantic_label)

        # Check common section for semantic_label
        if hasattr(robot_config, "common"):
            common = robot_config.common
            if hasattr(common, "semantic_label") and common.semantic_label:
                return str(common.semantic_label)
            # Also try dict-style access
            if isinstance(common, dict) and common.get("semantic_label"):
                return str(common["semantic_label"])

        # Fallback to robot type
        if robot_type:
            return robot_type.lower()

        # Final fallback to robot name
        return robot_name.lower()

    def _add_semantic_label(self, robot_cfg, semantic_label: str):
        """Apply semantic label to the robot root prim in env_0 only.
        Avoids touching descendants and avoids removing labels to prevent runtime resets.
        """
        try:
            stage = get_current_stage()
            if not stage:
                return
            base_prim_path = robot_cfg.robot.prim_path
            env_id = 0
            if "${ENV_REGEX_NS}" in base_prim_path:
                actual_prim_path = base_prim_path.replace(
                    "${ENV_REGEX_NS}", f"/World/envs/env_{env_id}"
                )
            elif re.search(r"/env_\d+/", base_prim_path):
                actual_prim_path = re.sub(
                    r"/env_\d+/", f"/env_{env_id}/", base_prim_path
                )
            else:
                parts = base_prim_path.split("/")
                env_inserted = False
                for i, part in enumerate(parts):
                    if part == "envs" and i + 1 < len(parts):
                        parts[i + 1] = f"env_{env_id}"
                        actual_prim_path = "/".join(parts)
                        env_inserted = True
                        break
                if not env_inserted:
                    return

            prim = stage.GetPrimAtPath(actual_prim_path)
            if prim and prim.IsValid():
                # Do not remove labels; just add on root to avoid subtree authoring
                add_labels(prim, [semantic_label])
        except Exception:
            pass

    def get_info(self):
        """Per-robot action term info plus ``max_eef_num`` / ``per_eef_dim`` from ``robot_cfgs``.

        ``per_eef_dim`` is ``eef_action`` total dim divided by ``max_eef_num`` when both are positive.
        """
        info = {}
        for robot_name, action_manager in self.action_managers.items():
            info[robot_name] = action_manager.get_info()
            rc = self.robot_cfgs.get(robot_name)
            if rc is None:
                raise RuntimeError(
                    f"RobotManager.get_info: robot_cfg missing for robot_name={robot_name!r}"
                )
            if not hasattr(rc, "planner"):
                raise RuntimeError(
                    f"RobotManager.get_info: robot_cfg for {robot_name!r} has no planner"
                )
            pl = rc.planner
            if not hasattr(pl, "max_eef_num"):
                raise RuntimeError(
                    f"RobotManager.get_info: planner for {robot_name!r} missing max_eef_num "
                    f"(define on the robot's PlannerCfg class)"
                )
            max_eef_num = int(pl.max_eef_num)
            eef_entry = info[robot_name].get("eef_action")
            eef_dim = int(eef_entry["action_dim"]) if eef_entry is not None else 0
            if max_eef_num < 0:
                raise ValueError(
                    f"RobotManager.get_info: {robot_name} max_eef_num must be >= 0, got {max_eef_num}"
                )
            if max_eef_num > 0 and eef_dim > 0 and eef_dim % max_eef_num != 0:
                raise ValueError(
                    f"RobotManager.get_info: {robot_name} eef_action action_dim={eef_dim} "
                    f"not divisible by max_eef_num={max_eef_num}"
                )
            per_eef_dim = (
                (eef_dim // max_eef_num) if max_eef_num > 0 and eef_dim > 0 else 0
            )
            info[robot_name]["max_eef_num"] = max_eef_num
            info[robot_name]["per_eef_dim"] = per_eef_dim
        return info
