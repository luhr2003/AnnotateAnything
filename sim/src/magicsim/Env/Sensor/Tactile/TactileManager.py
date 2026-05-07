from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple
from magicsim.Env.Environment.Isaac import IsaacRLEnv

if TYPE_CHECKING:
    from magicsim.Env.Robot.RobotManager import RobotManager
from omegaconf import DictConfig, OmegaConf
import torch
from isaacsim.core.utils.stage import get_current_stage

try:
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
    from tacex import GelSightSensor
except ImportError:
    raise ImportError(
        "Tacex is not installed. Please install Tacex to use TactileManager."
    )


class TactileManager:
    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        device: torch.device,
        robot_manager: RobotManager,
    ):
        print(
            "================================Initializing TactileManager================================"
        )
        self.num_envs = num_envs
        self.config = config
        self.device = device
        self.stage = get_current_stage()
        self.sim: IsaacRLEnv = None
        self.robot_manager = robot_manager
        self.sensors: Dict[str, GelSightSensor] = {}
        self.sensor_order: List[str] = []

    def initialize(self, sim: IsaacRLEnv):
        self.sim = sim
        for robot_name, tactile_cfg in self.config.items():
            robot_prim_path = self.robot_manager.robot_cfgs[robot_name].prim_path
            for sensor_name, sensor_cfg in tactile_cfg.items():
                sensor_key = f"{robot_name}_{sensor_name}"
                print(
                    f"TactileManager: attaching tactile sensor {sensor_key} to robot {robot_name} at {robot_prim_path}"
                )
                sensor_instance = self._create_sensor(robot_prim_path, sensor_cfg)
                self.sim.scene.sensors[sensor_key] = sensor_instance
                self.sensors[sensor_key] = sensor_instance
                self.sensor_order.append(sensor_key)

    def _create_sensor(
        self, robot_prim_path: str, sensor_cfg: DictConfig
    ) -> GelSightSensor:
        sensor_type = sensor_cfg.get("type")
        if sensor_type != "gelsight_mini":
            raise ValueError(f"Unsupported tactile sensor type: {sensor_type}")

        prim_link = sensor_cfg.get("prim_link")
        if prim_link is None:
            raise ValueError("Tactile sensor configuration requires 'prim_link'.")

        prim_path = f"{robot_prim_path}/{prim_link}"
        overrides = sensor_cfg.get("config")
        gelsight_cfg = GelSightMiniCfg(prim_path=prim_path)
        if overrides is not None:
            overrides_dict = OmegaConf.to_container(overrides, resolve=True)
            gelsight_cfg = self._apply_overrides(gelsight_cfg, overrides_dict)
        return GelSightSensor(gelsight_cfg)

    def _apply_overrides(self, cfg_obj, overrides: dict):
        if overrides is None:
            return cfg_obj
        simple_kwargs = {}
        nested_overrides = {}
        for key, value in overrides.items():
            if isinstance(value, dict) and hasattr(
                getattr(cfg_obj, key, None), "replace"
            ):
                nested_overrides[key] = value
            else:
                simple_kwargs[key] = value
        if simple_kwargs:
            cfg_obj = cfg_obj.replace(**simple_kwargs)
        for key, value in nested_overrides.items():
            nested_obj = getattr(cfg_obj, key)
            updated_nested = self._apply_overrides(nested_obj, value)
            cfg_obj = cfg_obj.replace(**{key: updated_nested})
        return cfg_obj

    def get_sensor_data(self) -> Tuple:
        return tuple(self.sensors[key].data for key in self.sensor_order)
