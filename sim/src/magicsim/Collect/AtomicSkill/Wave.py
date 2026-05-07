from typing import Any
import torch
from magicsim.Collect.AtomicSkill.Grasp import Grasp
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class Wave(Grasp):
    """
    Wave AtomicSkill: Grasp an object and then randomly wave/jitter it around.

    Combines Grasp functionality with jitter behavior:
    1. Execute Grasp phases: pre_grasp → grasp → close_gripper → retrieval
    2. After retrieval, perform random jittering around the retrieval pose
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        # Initialize Grasp first
        super().__init__(config, env, env_id, logger)

        # Wave-specific parameters
        self.jitter_steps_total: int = int(getattr(config, "jitter_steps", 300))
        self.jitter_translation_std: float = float(
            getattr(config, "jitter_translation_std", 0.05)
        )  # 5cm
        self.jitter_rotation_std: float = float(
            getattr(config, "jitter_rotation_std", 0.05)
        )  # ~0.05 rad

        # Wave phase tracking
        self._jitter_step_count: int = 0
        self._base_wave_pose: torch.Tensor | None = None  # Base pose for jittering

    def reset(self, action: list[Any]):
        """Reset Wave AtomicSkill - initialize Grasp and prepare for wave phase."""
        # Reset Grasp first
        super().reset(action)

        # Update command to Wave
        self.current_command = ["Wave", self.obj_type, self.obj_name, self.obj_id]

        # Reset wave-specific state
        self._jitter_step_count = 0
        self._base_wave_pose = None

    def refresh(self, action: list[Any]):
        """Refresh Wave command - update target object but keep current phase."""
        new_obj_type = action[1]
        new_obj_name = action[2]
        new_obj_id = action[3]
        new_command = ["Wave", new_obj_type, new_obj_name, new_obj_id]

        # Check if command changed
        command_changed = (
            self.current_command is None
            or self.current_command[1] != new_obj_type
            or self.current_command[2] != new_obj_name
            or self.current_command[3] != new_obj_id
        )

        self.obj_type = new_obj_type
        self.obj_name = new_obj_name
        self.obj_id = new_obj_id
        self.current_command = new_command

        # Reset phase only if command changed, otherwise keep current phase
        if command_changed or self.current_phase is None:
            self.current_phase = "pre_grasp"
            # Cancel any previous async job by bumping token and clearing job
            self._grasp_token += 1
            self._grasp_job = None
            self.grasp_pose = None
            self.pre_grasp_pose = None
            self.retrieval_grasp_pose = None
            self._base_wave_pose = None
            self._jitter_step_count = 0
            self.get_grasp_pose()

    def _get_eef_pose(self):
        """Get current 7D pose (pos+quat) of end effector for this env."""
        eef_pose = self.env.get_eef_pose([self.env_id])
        pos = eef_pose[0, :3]
        quat = eef_pose[0, 3:7]
        return pos, quat

    def _build_movl_action(self, pos: torch.Tensor, quat: torch.Tensor):
        """Helper to build MoveL action dict with [pos, quat, gripper_flag]."""
        return {
            "MoveL": (
                0,
                torch.cat(
                    [
                        pos,
                        quat,
                        torch.tensor([1.0], device=self.env.device),  # Gripper closed
                    ],
                    dim=0,
                ),
            )
        }

    def step(self):
        """Step Wave AtomicSkill - handle Grasp phases and wave jittering."""
        # Handle Grasp phases first
        if self.current_state == "failed":
            self.current_state = "failed"
            self.current_action = "Failed"
            return "Failed"

        # Ensure grasp poses are computed (async, non-blocking)
        if (
            self.grasp_pose is None
            or self.pre_grasp_pose is None
            or self.retrieval_grasp_pose is None
        ):
            self.get_grasp_pose()
            if self.current_state == "computing":
                self.current_action = None
                return None
            # If not computing anymore but still missing poses => failed
            if (
                self.grasp_pose is None
                or self.pre_grasp_pose is None
                or self.retrieval_grasp_pose is None
            ):
                self.current_state = "failed"
                self.current_action = "Failed"
                return "Failed"

        self.current_state = "running"

        # Handle Grasp phases
        if self.current_phase == "pre_grasp":
            target_pose_7d = self.pre_grasp_pose.to(device=self.env.device)
            self.current_action = {"MoveL": (0, target_pose_7d)}
            return {"MoveL": (0, target_pose_7d)}

        elif self.current_phase == "grasp":
            target_pose_7d = self.grasp_pose.to(device=self.env.device)
            self.current_action = {"MoveL": (0, target_pose_7d)}
            return {"MoveL": (0, target_pose_7d)}

        elif self.current_phase == "close_gripper":
            gripper_target = torch.tensor(
                [1.0], device=self.env.device, dtype=torch.float32
            )
            self.current_action = {"ParallelGripper": (0, gripper_target)}
            return {"ParallelGripper": (0, gripper_target)}

        elif self.current_phase == "retrieval":
            target_pose_7d = self.retrieval_grasp_pose.to(device=self.env.device)
            self.current_action = {"MoveL": (0, target_pose_7d)}
            return {"MoveL": (0, target_pose_7d)}

        elif self.current_phase == "wave":
            # Wave phase: random jittering around retrieval pose
            # Initialize base pose on first wave step
            if self._base_wave_pose is None:
                self._base_wave_pose = self.retrieval_grasp_pose.clone()

            base_pos = self._base_wave_pose[:3]
            base_quat = self._base_wave_pose[3:7]

            # Generate random translation jitter
            noise = torch.randn(3, device=self.env.device) * self.jitter_translation_std
            jitter_pos = base_pos + noise

            # Rotation jitter: add small noise to quaternion and normalize
            quat_noise = (
                torch.randn(4, device=self.env.device) * self.jitter_rotation_std
            )
            jitter_quat = base_quat + quat_noise
            jitter_quat = jitter_quat / torch.norm(jitter_quat).clamp_min(1e-6)

            self.current_action = self._build_movl_action(jitter_pos, jitter_quat)
            self._jitter_step_count += 1

            # Check if jittering is complete
            if self._jitter_step_count >= self.jitter_steps_total:
                self.current_phase = "done"

            return self.current_action

        elif self.current_phase == "done":
            # All phases completed, no more actions
            self.current_action = None
            return None
        else:
            # Unknown phase
            self.current_state = "failed"
            self.current_action = None
            return None

    def update(self, info):
        """Update Wave AtomicSkill state based on global planner feedback."""
        # During async compute, global planner might not exist yet
        if self.current_state == "computing":
            return {
                "atomic_skill_type": "Wave",
                "command": self.current_command,
                "action": None,
                "finished": False,
                "state": "computing",
                "truncated": 0,
                "phase": self.current_phase,
            }

        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "atomic_skill_type": "Wave",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            return {
                "type": "Wave",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state or "ready",
                "truncated": 0,
                "phase": self.current_phase,
            }

        if gp_info[self.env_id]["finished"]:
            # Global planner finished, move to next phase
            if self.current_phase == "pre_grasp":
                self.current_phase = "grasp"
                return {
                    "type": "Wave",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: moving to grasp",
                    "truncated": 0,
                    "phase": "grasp",
                }
            elif self.current_phase == "grasp":
                self.current_phase = "close_gripper"
                return {
                    "type": "Wave",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: closing gripper",
                    "truncated": 0,
                    "phase": "close_gripper",
                }
            elif self.current_phase == "close_gripper":
                self.current_phase = "retrieval"
                return {
                    "type": "Wave",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: retrieving",
                    "truncated": 0,
                    "phase": "retrieval",
                }
            elif self.current_phase == "retrieval":
                # Retrieval completed, start wave phase
                self.current_phase = "wave"
                self._base_wave_pose = self.retrieval_grasp_pose.clone()
                self._jitter_step_count = 0
                return {
                    "type": "Wave",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: waving",
                    "truncated": 0,
                    "phase": "wave",
                }
            elif self.current_phase == "wave":
                # Wave phase: continue generating jitter actions
                # Check if wave phase is complete (jitter_steps_total reached)
                # This check happens here because step() increments jitter_step_count
                if self._jitter_step_count >= self.jitter_steps_total:
                    self.current_phase = "done"
                    self.current_state = "finished"
                    return {
                        "type": "Wave",
                        "command": self.current_command,
                        "action": self.current_action,
                        "finished": True,
                        "state": self.current_state,
                        "truncated": 0,
                        "phase": "completed",
                    }
                else:
                    # Wave phase continues
                    return {
                        "type": "Wave",
                        "command": self.current_command,
                        "action": self.current_action,
                        "finished": False,
                        "state": f"running: waving ({self._jitter_step_count}/{self.jitter_steps_total})",
                        "truncated": 0,
                        "phase": "wave",
                    }
            elif self.current_phase == "done":
                # All phases completed
                self.current_state = "finished"
                return {
                    "type": "Wave",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "phase": "completed",
                }
        elif gp_info[self.env_id]["truncated"] == 1:
            # Global planner truncated, wave finished
            self.current_state = "truncated: env terminated first"
            return {
                "type": "Wave",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 2:
            # Global planner truncated, wave truncated
            self.current_state = "truncated: env truncated first"
            return {
                "type": "Wave",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 3:
            # Global planner failed
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "Wave",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "phase": self.current_phase,
            }
        else:
            # Global planner running
            # For wave phase, completion is already checked above when gp_info["finished"] is True
            # Here we just return running state
            self.current_state = "running"
            return {
                "type": "Wave",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "phase": self.current_phase,
            }
