"""
Lift atomic skill: bimanual box-squeeze driven by the target object's
bounding box.

Unlike :class:`DexGrasp` (which selects among annotated paired grasps via
IK goalset), :class:`Lift` asks the env for the target's local AABB
half-extents + current world pose, derives three world-frame wrist targets
(pre-grasp parked outside the ``±y`` faces, squeeze pressed inward past the
faces, lift the squeeze target raised by ``lift_height``), and drives each
phase through :class:`MobileMoveL` with ``hand_id=-1`` (paired dual-arm).

Hands stay fully closed throughout — the squeeze is forearm-driven.

Env contract: the underlying :class:`TaskBaseEnv` must expose
``get_target_bbox_half_extents(env_id, obj_name, obj_id) -> (hx, hy, hz)``
and ``get_target_world_pose(env_id, obj_name, obj_id) -> tensor [7]``
(same shape / convention as :class:`LocoLiftEnv`).

Phases: ``pre_grasp -> squeeze -> lift``
"""

from typing import Any, List

import torch
from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


# Default fully-closed hand joints for the Dex3.
# Joint order per ``G1.eef_action['joint_pos']`` groups (G1.py:851 right,
# G1.py:864 left): index_0, index_1, middle_0, middle_1, thumb_0, thumb_1,
# thumb_2. Right flexes with positive; left is the mirror.
# :class:`MultipleJointPositionToLimitsAction` clamps to soft limits.
_DEFAULT_RIGHT_CLOSE: List[float] = [1.5, 1.7, 1.5, 1.7, 0.0, -0.7, -0.7]
_DEFAULT_LEFT_CLOSE: List[float] = [-1.5, -1.7, -1.5, -1.7, 0.0, 0.7, 0.7]


class Lift(AtomicSkill):
    """Bimanual box squeeze via ``MobileMoveL`` + bbox-derived targets."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.hand_id = -1  # bimanual

        # Geometry / phase knobs (meters, ratios).
        self.gap = float(getattr(config, "gap", 0.02))
        self.pre_gap = float(getattr(config, "pre_gap", 0.15))
        self.forward_ratio = float(getattr(config, "forward_ratio", 1.5))
        self.down_ratio = float(getattr(config, "down_ratio", 0.1))
        self.lift_height = float(getattr(config, "lift_height", 0.2))

        self.hand_joint_dim = int(getattr(config, "hand_joint_dim", 7))
        self.debug = bool(getattr(config, "debug", False))

        right_close = list(getattr(config, "right_close_hand", _DEFAULT_RIGHT_CLOSE))
        left_close = list(getattr(config, "left_close_hand", _DEFAULT_LEFT_CLOSE))
        self._right_close_cpu = torch.tensor(right_close, dtype=torch.float32)
        self._left_close_cpu = torch.tensor(left_close, dtype=torch.float32)

        # Phase state (populated in reset after the env provides bbox/pose).
        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None
        self.current_phase: str | None = None
        self.r_pre = self.l_pre = None
        self.r_grasp = self.l_grasp = None
        self.r_lift = self.l_lift = None

        self.planner_manager = None

    # ------------------------------------------------------------------ helpers
    # 90° wrist roll around world ±x → knuckle face (back of fist) faces
    # the bin instead of the thumb. Right and left mirrored.
    _WRIST_R = 0.7071067811865476
    _RIGHT_WRIST_QUAT = (_WRIST_R, +_WRIST_R, 0.0, 0.0)  # 90° about +x
    _LEFT_WRIST_QUAT = (_WRIST_R, -_WRIST_R, 0.0, 0.0)  # 90° about -x

    def _right_wrist_quat(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._RIGHT_WRIST_QUAT, device=device, dtype=torch.float32)

    def _left_wrist_quat(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._LEFT_WRIST_QUAT, device=device, dtype=torch.float32)

    def _compute_targets(self) -> bool:
        """Fill ``r_pre/l_pre/r_grasp/l_grasp/r_lift/l_lift`` from the env's bbox + world pose."""
        if not hasattr(self.env, "get_target_bbox_half_extents") or not hasattr(
            self.env, "get_target_world_pose"
        ):
            self.current_state = (
                "failed: env lacks get_target_bbox_half_extents/get_target_world_pose"
            )
            return False

        half = self.env.get_target_bbox_half_extents(
            env_id=self.env_id, obj_name=self.obj_name, obj_id=int(self.obj_id)
        )
        if half is None:
            self.current_state = "failed: bbox unavailable"
            return False

        pose = self.env.get_target_world_pose(
            env_id=self.env_id, obj_name=self.obj_name, obj_id=int(self.obj_id)
        )
        if pose is None:
            self.current_state = "failed: target world pose unavailable"
            return False

        device = self.env.device
        bin_pos = pose[:3].to(device)
        bin_quat = pose[3:7].to(device)

        hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
        x_local = self.forward_ratio * hx
        z_local = -self.down_ratio * hz
        right_quat = self._right_wrist_quat(device)
        left_quat = self._left_wrist_quat(device)

        def _pack(y_local: float):
            # Robot faces +x → right hand on -y, left hand on +y. Variable
            # r_* must land on -y so it goes into the right_hand task slot
            # (Pink IK: task[0]=right, task[1]=left in G1.py:135). Inverting
            # this would cross the arms and IK fails immediately.
            # Per-hand wrist quat (90° roll, mirrored) so knuckles face inward.
            r_local = torch.cat(
                [
                    torch.tensor([x_local, -y_local, z_local], device=device),
                    right_quat,
                ]
            )
            l_local = torch.cat(
                [
                    torch.tensor([x_local, +y_local, z_local], device=device),
                    left_quat,
                ]
            )
            return (
                RigidObject.transform_pose_to_world(r_local, bin_pos, bin_quat),
                RigidObject.transform_pose_to_world(l_local, bin_pos, bin_quat),
            )

        self.r_pre, self.l_pre = _pack(hy + self.pre_gap)
        self.r_grasp, self.l_grasp = _pack(hy - self.gap)
        self.r_lift = self.r_grasp.clone()
        self.r_lift[2] += self.lift_height
        self.l_lift = self.l_grasp.clone()
        self.l_lift[2] += self.lift_height

        if self.debug:
            print(
                f"[Lift] env_id={self.env_id} half=({hx:.3f},{hy:.3f},{hz:.3f}) "
                f"x_local={x_local:.3f} z_local={z_local:.3f} "
                f"pre_gap={self.pre_gap} gap={self.gap} lift={self.lift_height}"
            )
        return True

    def _build_action(
        self, right_arm: torch.Tensor, left_arm: torch.Tensor
    ) -> torch.Tensor:
        """28D dual-arm eef: ``[right_arm(7), left_arm(7), right_hand(7), left_hand(7)]``.

        Hand slot order matches ``G1.eef_action['joint_pos']`` (right first,
        left second). Hands stay fully closed throughout — fingers curl in
        so the bin can't squirt upward when the forearms compress its sides.
        """
        device = self.env.device
        return torch.cat(
            [
                right_arm.to(device),
                left_arm.to(device),
                self._right_close_cpu.to(device),
                self._left_close_cpu.to(device),
            ],
            dim=0,
        )

    def _placeholder_action(self) -> dict:
        device = self.env.device
        eef = torch.full(
            (14 + 2 * self.hand_joint_dim,),
            torch.nan,
            device=device,
            dtype=torch.float32,
        )
        return {"MobileMoveL": ((self.robot_id, -1, 3), eef)}

    # ------------------------------------------------------------------ reset / refresh
    def reset(self, action: List[Any]):
        # action: [Lift, robot_id, obj_type, obj_name, obj_id]
        self.robot_id = int(action[1])
        self.hand_id = -1
        self.obj_type = action[2]
        self.obj_name = action[3]
        self.obj_id = action[4]

        self.current_state = "ready"
        self.current_command = list(action)
        self.current_phase = "pre_grasp"

        self.planner_manager = getattr(self.env.scene, "planner_manager", None)
        if self.planner_manager is None:
            self.current_state = "failed: planner_manager not available"
            return
        if self.obj_name is not None:
            # Only ignore the bin (squeeze pushes wrist past ±y face). Keep
            # the table in the obstacle set — wrist target must be raised
            # high enough (negative ``down_ratio``) that the Dex3 palm
            # sphere clears the table top.
            self.planner_manager.update_obstacles(
                obstacle_avoidance_path_list=["dynamic"],
                env_ids=[self.env_id],
                obstacle_ignore_path_list=[self.obj_name],
            )
            self._compute_targets()

        if self.debug:
            print(
                f"[Lift] env_id={self.env_id} reset robot_id={self.robot_id} "
                f"obj={self.obj_type}/{self.obj_name}/{self.obj_id} phase=pre_grasp"
            )

    def refresh(self, action: List[Any]):
        new_cmd = list(action)
        if self.current_command != new_cmd or self.current_phase is None:
            self.reset(action)

    # ------------------------------------------------------------------ step / update
    def step(self):
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        # Placeholder until the target object is bound (init phase from the
        # LocoLift task wrapper).
        if self.obj_name is None or self.obj_id is None:
            self.current_action = self._placeholder_action()
            return self.current_action

        if self.r_pre is None or self.r_grasp is None or self.r_lift is None:
            if not self._compute_targets():
                self.current_action = "Failed"
                return "Failed"

        # MobileMoveL planner_mode (see MobileMoveL.py:28-39):
        #   pre_grasp: mode=3 → free base + MotionGen (navigate to bin first)
        #   squeeze:   mode=0 → locked base, no IK, no MotionGen
        #               (direct single-step arm target — bin is in obstacle
        #                set so MotionGen would refuse to plan into it)
        #   lift:      mode=0 → locked base, no IK, no MotionGen
        #               (vertical lift after squeeze; same reason)
        if self.current_phase == "pre_grasp":
            eef = self._build_action(self.r_pre, self.l_pre)
            mode = 3
        elif self.current_phase == "squeeze":
            eef = self._build_action(self.r_grasp, self.l_grasp)
            mode = 0
        elif self.current_phase == "lift":
            eef = self._build_action(self.r_lift, self.l_lift)
            mode = 0
        else:
            self.current_state = "failed"
            self.current_action = "Failed"
            return "Failed"

        self.current_state = "running"
        self.current_action = {"MobileMoveL": ((self.robot_id, -1, mode), eef)}
        return self.current_action

    def update(self, info: dict) -> dict:
        base = {
            "atomic_skill_type": "Lift",
            "command": self.current_command,
            "action": self.current_action,
            "phase": self.current_phase,
        }
        if self.current_state == "failed":
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            return {
                **base,
                "finished": False,
                "state": self.current_state or "ready",
                "truncated": 0,
            }
        env_gp = gp_info[self.env_id]
        trunc = env_gp.get("truncated", 0)
        if trunc == 1:
            self.current_state = "truncated: env terminated"
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        if trunc == 2:
            self.current_state = "truncated: env truncated"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        if trunc == 3:
            self.current_state = "failed: global planner failed"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        if trunc == 5:
            return {
                **base,
                "finished": False,
                "state": self.current_state or "running",
                "truncated": 0,
            }

        if env_gp.get("finished", False):
            if self.current_phase == "pre_grasp":
                # Re-snapshot bin pose right before squeeze: the bin may
                # have shifted slightly during pre_grasp (or settled further
                # under gravity). Targets at reset time were captured for
                # navigation; refresh squeeze + lift targets against the
                # bin's actual current pose so the forearms close on it.
                self._compute_targets()
                self.current_phase = "squeeze"
                if self.debug:
                    print(
                        f"[Lift] env_id={self.env_id} phase=squeeze (targets refreshed)"
                    )
                return {
                    **base,
                    "finished": False,
                    "state": "running: squeeze",
                    "truncated": 0,
                    "phase": "squeeze",
                }
            if self.current_phase == "squeeze":
                self.current_phase = "lift"
                if self.debug:
                    print(f"[Lift] env_id={self.env_id} phase=lift")
                return {
                    **base,
                    "finished": False,
                    "state": "running: lift",
                    "truncated": 0,
                    "phase": "lift",
                }
            if self.current_phase == "lift":
                self.current_state = "finished"
                if self.debug:
                    print(f"[Lift] env_id={self.env_id} phase=completed")
                return {
                    **base,
                    "finished": True,
                    "state": "finished",
                    "truncated": 0,
                    "phase": "completed",
                }

        return {**base, "finished": False, "state": "running", "truncated": 0}
