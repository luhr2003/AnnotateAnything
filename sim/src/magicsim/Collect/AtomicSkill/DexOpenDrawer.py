"""DexOpenDrawer atomic skill — Franka + Xhand variant of :class:`OpenDrawer`.

Same five-phase state machine as the parent (approach → grasp_handle →
close_gripper → pull → release) but every motion command is the 19-D
``[arm_pose (7) | hand_joints (12)]`` action expected by the dexterous robot
(see :class:`magicsim.Collect.AtomicSkill.DexGrasp` for the same pattern).
The trajectory comes from the ``xhand_open_by_handle_trajectory`` annotation
loaded by :meth:`DexOpenDrawerEnv.get_drawer_trajectories`; finger joint
targets recorded with the trajectory are used verbatim during pull, and the
first-waypoint joint targets are reused as the "closed" pose during the
``close_gripper`` phase.
"""

from typing import Any
import torch

from magicsim.Collect.AtomicSkill.OpenDrawer import OpenDrawer
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


ARM_POSE_DIM = 7
FINGER_JOINT_DIM = 12
DEX_ACTION_DIM = ARM_POSE_DIM + FINGER_JOINT_DIM


class DexOpenDrawer(OpenDrawer):
    """Open-drawer skill for the Franka + Xhand robot."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        # Default annotation for the dexterous variant — overridable via config.
        traj_cfg = getattr(config, "trajectory", None)
        if traj_cfg is None or not getattr(traj_cfg, "annotation_name", None):
            self._annotation_name = "xhand_open_by_handle_trajectory"
        # Cached pull trajectory ([N, 19]); reuses parent's identity check.
        self._pull_traj_19d_cached: torch.Tensor | None = None
        # Closed-hand finger joints derived from trajectory[0]; defaults to zeros.
        self._closed_hand_joints: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # IK goalset selection — slice 19-D candidates to 7-D for the IK API.
    # ------------------------------------------------------------------ #

    def _start_ik_job(self, candidate_poses: torch.Tensor, candidate_keys: list[str]):
        # Parent stacks the first waypoint of each trajectory; for dex these are
        # 19-D rows. The IK service only consumes pose (7), so slice here.
        if candidate_poses.ndim == 2 and candidate_poses.shape[1] >= ARM_POSE_DIM:
            pose_only = candidate_poses[:, :ARM_POSE_DIM].contiguous()
        else:
            pose_only = candidate_poses
        super()._start_ik_job(pose_only, candidate_keys)

    def _select_trajectory(self, key: str) -> None:
        """Cache pose-only handle and finger-joint targets after IK selection."""
        traj = self.all_raw_trajectories[key]
        if traj.shape[1] >= DEX_ACTION_DIM:
            self._closed_hand_joints = traj[0, ARM_POSE_DIM:DEX_ACTION_DIM].clone()
        else:
            self._closed_hand_joints = torch.zeros(
                FINGER_JOINT_DIM, device=traj.device, dtype=traj.dtype
            )

    def get_handle_pose(self):
        result = super().get_handle_pose()
        # When parent finishes IK and assigns ``selected_raw_trajectory``, sync
        # our cached finger-joint targets (handle_pose stays 7-D from parent).
        if (
            self.selected_raw_trajectory is not None
            and self._closed_hand_joints is None
            and self.selected_trajectory_key is not None
        ):
            self._select_trajectory(self.selected_trajectory_key)
        return result

    def reset(self, action: list[Any]):
        self._pull_traj_19d_cached = None
        self._closed_hand_joints = None
        super().reset(action)

    def refresh(self, action: list[Any]):
        super().refresh(action)
        # Parent may have wiped trajectory state; mirror here.
        if self.selected_raw_trajectory is None:
            self._closed_hand_joints = None
            self._pull_traj_19d_cached = None

    # ------------------------------------------------------------------ #
    # 19-D action assembly per phase.
    # ------------------------------------------------------------------ #

    def _open_hand(self) -> torch.Tensor:
        return torch.zeros(
            FINGER_JOINT_DIM, device=self.env.device, dtype=torch.float32
        )

    def _closed_hand(self) -> torch.Tensor:
        if self._closed_hand_joints is None:
            return self._open_hand()
        return self._closed_hand_joints.to(self.env.device, dtype=torch.float32)

    def _arm_plus_hand(
        self, arm_pose: torch.Tensor, hand_joints: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [arm_pose.to(self.env.device, dtype=torch.float32), hand_joints], dim=0
        )

    _step_count = 0

    def step(self):
        DexOpenDrawer._step_count += 1
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        if self.handle_pose is None or self.pre_grasp_pose is None:
            self.get_handle_pose()
            if self.current_state == "computing":
                self.current_action = None
                return None
            if self.handle_pose is None or self.pre_grasp_pose is None:
                self.current_state = "failed"
                self.current_action = "Failed"
                return "Failed"

        self.current_state = "running"
        robot_id = self.robot_id
        hand_id = self.hand_id
        move_key = self._move_planner_key
        servo_key = self._servo_planner_key

        if self.current_phase == "approach":
            target = self._arm_plus_hand(self.pre_grasp_pose, self._open_hand())
            self.current_action = {move_key: ((robot_id, hand_id, -1), target)}
            return self.current_action

        if self.current_phase == "grasp_handle":
            target = self._arm_plus_hand(self.handle_pose, self._open_hand())
            self.current_action = {move_key: ((robot_id, hand_id, -1), target)}
            return self.current_action

        if self.current_phase == "close_gripper":
            # Hold arm pose, drive fingers to recorded grasp targets.
            target = self._arm_plus_hand(self.handle_pose, self._closed_hand())
            self.current_action = {move_key: ((robot_id, hand_id, 0), target)}
            return self.current_action

        if self.current_phase == "pull":
            if self.selected_raw_trajectory is not None:
                if self._pull_traj_19d_cached is None:
                    if self.pull_trajectory is None:
                        self.pull_trajectory = self._compute_pull_trajectory()
                    traj = self.pull_trajectory.to(
                        device=self.env.device, dtype=torch.float32
                    )
                    if traj.shape[1] != DEX_ACTION_DIM:
                        # Pad finger joints if a 7-D trajectory slipped through.
                        if traj.shape[1] == ARM_POSE_DIM:
                            joints = (
                                self._closed_hand()
                                .unsqueeze(0)
                                .expand(traj.shape[0], -1)
                            )
                            traj = torch.cat([traj, joints], dim=1)
                    self._pull_traj_19d_cached = traj.contiguous()
                self.current_action = {
                    servo_key: ((robot_id, hand_id, 0), self._pull_traj_19d_cached)
                }
            else:
                target = self._arm_plus_hand(self.pulled_pose, self._closed_hand())
                self.current_action = {move_key: ((robot_id, hand_id, 0), target)}
            return self.current_action

        if self.current_phase == "release":
            target = self._arm_plus_hand(self.handle_pose, self._open_hand())
            self.current_action = {move_key: ((robot_id, hand_id, 0), target)}
            return self.current_action

        self.current_state = "failed"
        self.current_action = None
        return None

    def update(self, info):
        result = super().update(info)
        if isinstance(result, dict):
            result["atomic_skill_type"] = "DexOpenDrawer"
            if "type" in result:
                result["type"] = "DexOpenDrawer"
        return result
