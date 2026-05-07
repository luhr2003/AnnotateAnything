"""
Dehatch AtomicSkill: stand up, retract hands, then retract backward.

Action format: ["Dehatch", robot_id, hand_id]

Three phases, each dispatched to a different GlobalPlanner:

1. **standup** → MobileServoL: pelvis rises while EEFs keep relative offset.
   (Optional — skipped if ``skip_standup`` is set.)
2. **retract_hands** → MobileMoveL (hand_id=-1, planner_mode=1): fixed-base
   MotionGen (no IK) moves both arms to rest poses.
3. **retract_base** → MobileServoL: pelvis walks backward, EEFs at rest poses.

The ``dehatch_strategy`` function is resolved from the robot's planner config
(``robot_cfg.planner.dehatch_strategy``), so each robot type can provide its own
implementation.
"""

from typing import Any, Callable, List, Optional

from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv

# Phase order
_PHASES = ("standup", "retract_hands", "retract_base")
_PHASES_NO_STANDUP = ("retract_hands", "retract_base")


class Dehatch(AtomicSkill):
    """
    Multi-phase dehatch: stand up → retract hands (MotionGen) → retract base.

    Resolves ``dehatch_strategy`` from the robot's planner config at reset time.
    Each phase queries the strategy for ``(global_planner_key, data)`` and forwards
    to the GlobalPlannerManager.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.hand_id = int(getattr(config, "hand_id", -1))
        self.retract_distance = float(getattr(config, "retract_distance", 0.5))
        self.num_standup_steps = int(getattr(config, "num_standup_steps", 50))
        self.num_retract_steps = int(getattr(config, "num_retract_steps", 100))
        self.skip_standup = bool(getattr(config, "skip_standup", False))
        self._dehatch_strategy: Optional[Callable] = None
        self._phase_idx: int = 0
        self._phases = _PHASES
        self._gp_key: Optional[str] = None
        self._gp_data: Any = None

    # ------------------------------------------------------------------ #
    # Robot state / config helpers
    # ------------------------------------------------------------------ #

    def _get_robot_name_list(self) -> List[str]:
        return list(self.env.scene.robot_manager.robots.keys())

    def _get_robot_name(self) -> str:
        name_list = self._get_robot_name_list()
        if self.robot_id < len(name_list):
            return name_list[self.robot_id]
        return name_list[0] if name_list else ""

    def _get_robot_state(self) -> dict:
        robot_states = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        robot_name = self._get_robot_name()
        if isinstance(robot_states, dict):
            if robot_name and robot_name in robot_states:
                return robot_states[robot_name]
            return next(iter(robot_states.values()))
        return robot_states

    def _get_dehatch_strategy(self) -> Optional[Callable]:
        robot_name = self._get_robot_name()
        robot_cfg = self.env.scene.robot_manager.robot_cfgs.get(robot_name, None)
        if robot_cfg is not None and hasattr(robot_cfg, "planner"):
            return getattr(robot_cfg.planner, "dehatch_strategy", None)
        return None

    # ------------------------------------------------------------------ #
    # Phase management
    # ------------------------------------------------------------------ #

    @property
    def _current_phase(self) -> str:
        return self._phases[self._phase_idx]

    def _query_phase(self) -> None:
        """Query the strategy for the current phase and cache the result."""
        robot_state = self._get_robot_state()
        self._gp_key, self._gp_data = self._dehatch_strategy(
            phase=self._current_phase,
            robot_state=robot_state,
            retract_distance=self.retract_distance,
            num_standup_steps=self.num_standup_steps,
            num_retract_steps=self.num_retract_steps,
            env_id=self.env_id,
        )

    def _build_action(self) -> dict:
        """Build the GlobalPlanner action dict for the current phase."""
        if self._gp_key == "MobileMoveL":
            # MobileMoveL: hand_id=-1 (dual), planner_mode=1 (fixed base, force MotionGen, no IK)
            return {
                self._gp_key: (
                    (self.robot_id, -1, 1),
                    self._gp_data,
                )
            }
        else:
            # MobileServoL: pass trajectory directly
            return {
                self._gp_key: (
                    (self.robot_id, self.hand_id, 0),
                    self._gp_data,
                )
            }

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, action: list):
        """
        action: ["Dehatch", robot_id, hand_id]
        """
        self.robot_id = int(action[1])
        self.hand_id = int(action[2]) if len(action) > 2 else -1
        self.current_command = list(action)
        self.current_state = "ready"

        self._dehatch_strategy = self._get_dehatch_strategy()
        if self._dehatch_strategy is None:
            robot_name = self._get_robot_name()
            raise RuntimeError(
                f"No dehatch_strategy configured for robot '{robot_name}'. "
                f"Set it on the robot's planner config (e.g. G1PlannerCfg.dehatch_strategy)."
            )

        self._phases = _PHASES_NO_STANDUP if self.skip_standup else _PHASES
        self._phase_idx = 0
        self._query_phase()

    def refresh(self, action: list):
        self.robot_id = int(action[1])
        self.hand_id = int(action[2]) if len(action) > 2 else -1
        self.current_command = list(action)

    def step(self):
        self.current_state = "running"
        self.current_action = self._build_action()
        return self.current_action

    def update(self, info: Any) -> dict:
        base = {
            "type": "Dehatch",
            "command": self.current_command,
            "action": self.current_action,
            "phase": self._current_phase,
        }

        gp_info = info.get("global_planner_info", [None] * (self.env_id + 1))
        if gp_info[self.env_id] is None:
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }

        env_gp = gp_info[self.env_id]
        truncated = env_gp.get("truncated", 0)

        if truncated == 1:
            self.current_state = "finished"
            return {
                **base,
                "finished": True,
                "state": "finished",
                "truncated": truncated,
            }
        if truncated == 2:
            self.current_state = "truncated: env truncated"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        if truncated == 3:
            self.current_state = "failed: global planner failed"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        # Planner swap (truncated=5): not a real failure
        if truncated == 5:
            return {
                **base,
                "finished": False,
                "state": self.current_state or "running",
                "truncated": 0,
            }

        if env_gp.get("finished", False):
            # Advance to next phase
            if self._phase_idx < len(self._phases) - 1:
                self._phase_idx += 1
                self._query_phase()
                return {
                    **base,
                    "finished": False,
                    "state": f"running: {self._current_phase}",
                    "truncated": 0,
                    "phase": self._current_phase,
                }
            else:
                # All phases complete
                self.current_state = "finished"
                return {**base, "finished": True, "state": "finished", "truncated": 0}

        return {**base, "finished": False, "state": "running", "truncated": 0}
