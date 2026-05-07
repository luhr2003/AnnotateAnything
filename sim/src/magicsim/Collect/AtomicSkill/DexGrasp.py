"""
DexGrasp atomic skill for dexterous hand grasping (single-hand).

Mimics Grasp.py structure. IK runs only on coarse_grasp poses; once selected,
the full candidate (coarse/final) is used for subsequent phases.

Phases:
  1. pre_grasp    — arm backed off from coarse_grasp; hand open
  2. coarse_grasp — arm to coarse pose; hand open
  3. fine_grasp   — arm to final pose; hand open (precise positioning)
  4. final_grasp  — hand closes to final joints
  5. retrieval    — arm lifts z-up; hand keeps final joints

When viz_grasp is True, each phase draws axes for that phase's target pose
(pre / coarse / final / retrieval), once when the phase starts.

For bimanual / dual-hand dexterous grasping see :class:`BiDexGrasp`.
"""

from typing import Any
import torch

from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Planner.PlannerManager import PlannerManager
from magicsim.Task.LocoManip.Env.Test.TestLocoGraspEnv import visualize_grasp_pose
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix


def _to_tensor(x, dtype=torch.float32, device=None):
    """Convert to tensor; use detach().clone() when already a tensor to avoid UserWarning."""
    if isinstance(x, torch.Tensor):
        out = x.detach().clone().to(dtype=dtype)
        return out.to(device) if device is not None else out
    return torch.tensor(x, dtype=dtype, device=device)


def _extract_candidates_from_parts(parts: dict, part_name: str | None) -> list:
    """Extract candidate list from parts dict; if part_name given use that part, else all."""
    out = []
    if part_name and part_name in parts and isinstance(parts[part_name], list):
        out.extend(parts[part_name])
    if not out:
        for v in parts.values():
            if isinstance(v, list):
                out.extend(v)
    return out


class DexGrasp(AtomicSkill):
    """
    Single-hand dexterous grasp (e.g. Franka + XHand).
    IK only for coarse_grasp; rest uses annotated poses/joints directly.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_phase = (
            None  # pre_grasp, coarse_grasp, fine_grasp, final_grasp, retrieval
        )
        self.coarse_pose = None
        self.fine_pose = None
        self.final_pose = None
        self.pre_grasp_pose = None
        self.retrieval_pose = None
        self.coarse_joints = None
        self.fine_joints = None
        self.final_joints = None
        self.grasp_pose_type = None
        self.functional_grasp = True
        self.part = None

        self.pre_grasp_offset = getattr(config, "pre_grasp_offset", 0.05)
        self.retrieval_offset = getattr(config, "retrieval_offset", 0.2)
        self.mobile = getattr(config, "mobile", False)
        self.hand_type = getattr(config, "hand_type", "xhand")
        # hand_joint_dim: from config, or hand_joint_dim_map[hand_type], default 7 for dex3_1, 12 for xhand
        hand_joint_dim_map = getattr(
            config, "hand_joint_dim_map", {"xhand": 12, "dex3_1": 7}
        )
        self.hand_joint_dim = getattr(
            config, "hand_joint_dim", hand_joint_dim_map.get(self.hand_type, 7)
        )
        self.pre_grasp_z_threshold = float(
            getattr(config, "pre_grasp_z_threshold", 0.4)
        )
        self._pregrasp_use_retract = False
        # Placeholder all-NaN action: which GP + header mode (mobile pre_grasp uses 3)
        self._placeholder_gp_key = "MobileMoveL" if self.mobile else "MoveL"
        self._placeholder_mode = 3 if self.mobile else -1
        self.debug = getattr(config, "debug", False)
        self.viz_grasp = getattr(config, "viz_grasp", True)
        self.reactive = getattr(config, "reactive", True)

        self._grasp_job: dict | None = None
        self._grasp_token: int = 0
        self._selected_grasp_idx: int = -1
        self._coarse_grasp_pose_updated = (
            False  # Only update once per coarse_grasp when reactive
        )
        self._fine_grasp_pose_updated = False
        self._final_grasp_pose_updated = False
        self._last_viz_phase = (
            None  # Visualize grasp axes once per phase when viz_grasp
        )
        self.robot_id = 0
        self.hand_id = 0
        self.robot_name = None
        self.ik_server = None
        self.planner_manager: PlannerManager | None = None

    def _is_placeholder_target(self, arm_targets: torch.Tensor) -> bool:
        """Check if arm targets are all NaN (placeholder during init phase)."""
        return torch.isnan(arm_targets).all()

    def _build_placeholder_action(self) -> dict:
        """All-NaN target so MoveL/MobileMoveL skips IK until grasp pose is ready."""
        dev = self.env.device
        eef = torch.full(
            (7 + self.hand_joint_dim,),
            torch.nan,
            device=dev,
            dtype=torch.float32,
        )
        return {
            self._placeholder_gp_key: (
                (self.robot_id, self.hand_id, self._placeholder_mode),
                eef,
            )
        }

    def _move_l_mode_for_phase(self, phase: str) -> int:
        """MoveL header: IK path for pre/coarse, direct for later phases."""
        if phase in ("pre_grasp", "coarse_grasp"):
            return -1
        return 0

    def _mobile_mode_for_phase(self, phase: str) -> int:
        """Mobile: pre_grasp always uses header mode 3; other phases unchanged."""
        if phase == "pre_grasp":
            return 3
        if phase == "coarse_grasp":
            return -1
        if phase in ("fine_grasp", "final_grasp"):
            return 0
        if phase == "retrieval":
            return 3
        return -1

    def _mobile_planner_key_for_phase(self, phase: str) -> str:
        if phase == "pre_grasp" and self._pregrasp_use_retract:
            return "RetractMoveL"
        return "MobileMoveL"

    def _build_motion_action(
        self, arm_pose: torch.Tensor, hand_joints: torch.Tensor, phase: str
    ) -> dict:
        eef_action = torch.cat([arm_pose, hand_joints], dim=0)
        if not self.mobile:
            return {
                "MoveL": (
                    (
                        self.robot_id,
                        self.hand_id,
                        self._move_l_mode_for_phase(phase),
                    ),
                    eef_action,
                )
            }
        key = self._mobile_planner_key_for_phase(phase)
        return {
            key: (
                (
                    self.robot_id,
                    self.hand_id,
                    self._mobile_mode_for_phase(phase),
                ),
                eef_action,
            )
        }

    def _get_planner_manager(self):
        pm = getattr(self.env.scene, "planner_manager", None)
        if pm is None:
            raise RuntimeError("PlannerManager not available in the environment.")
        return pm

    def _resolve_robot_name(self) -> str:
        if self.robot_name is not None:
            return self.robot_name
        robot_manager = getattr(self.env.scene, "robot_manager", None)
        if robot_manager is not None:
            robot_dict = getattr(robot_manager, "robots", None)
            if isinstance(robot_dict, dict) and robot_dict:
                self.robot_name = next(iter(robot_dict.keys()))
                return self.robot_name
        raise RuntimeError("Unable to resolve robot_name.")

    def _get_robot_state(self) -> dict:
        robot_states = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        if isinstance(robot_states, dict):
            name = self._resolve_robot_name()
            return robot_states.get(name, next(iter(robot_states.values())))
        return robot_states

    def _submit_goalset(self, grasp_pose_list: torch.Tensor):
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        target = self.pack_single_arm_goalset(grasp_pose_list)
        is_dual_ik = getattr(self.ik_server, "dual_mode", False)
        if is_dual_ik:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=robot_states_dict,
                mode="goalset",
                lock_base=False,  # Check fixed-base reachability first (like MobileMoveL)
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=robot_states_dict,
                mode="goalset",
            )
        return self.ik_server.submit_ik(req)

    def reset(self, action: list[Any]):
        # action: [DexGrasp, robot_id, hand_id, obj_type, obj_name, obj_id, optional functional_grasp, optional part]
        # Backward compat: if len(action)==6 (old format), use robot_id=0, hand_id from config or 0
        if len(action) >= 8:
            self.robot_id = int(action[1])
            self.hand_id = int(action[2])
            self.obj_type = action[3]
            self.obj_name = action[4]
            self.obj_id = action[5]
            self.functional_grasp = action[6]
            self.part = action[7]
        else:
            self.robot_id = int(getattr(self.config, "robot_id", 0))
            self.hand_id = int(getattr(self.config, "hand_id", 0))
            self.obj_type = action[1]
            self.obj_name = action[2]
            self.obj_id = action[3]
            if len(action) >= 6:
                self.functional_grasp = action[4]
                self.part = action[5]
            else:
                self.functional_grasp = getattr(self.config, "functional_grasp", True)
                self.part = getattr(self.config, "functional_part", None)

        self.current_state = "ready"
        self.current_command = [
            "DexGrasp",
            self.robot_id,
            self.hand_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.functional_grasp,
            self.part,
        ]
        self.current_phase = "pre_grasp"
        if self.debug:
            print(f"[DexGrasp] env_id={self.env_id} phase=pre_grasp (reset)")

        self.coarse_pose = None
        self.fine_pose = None
        self.final_pose = None
        self.pre_grasp_pose = None
        self.retrieval_pose = None
        self.coarse_joints = None
        self.fine_joints = None
        self.final_joints = None
        self.grasp_pose_type = None
        self._grasp_token += 1
        self._grasp_job = None
        self._selected_grasp_idx = -1
        self._coarse_grasp_pose_updated = False
        self._fine_grasp_pose_updated = False
        self._final_grasp_pose_updated = False
        self._last_viz_phase = None
        self._pregrasp_use_retract = False

        self.planner_manager = self._get_planner_manager()
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            raise RuntimeError("IKServer not available in PlannerManager.")
        robot_name_list = list(ik_dict.keys())
        if self.robot_id < 0 or self.robot_id >= len(robot_name_list):
            raise RuntimeError(
                f"DexGrasp: robot_id {self.robot_id} out of range for "
                f"robot_name_list={robot_name_list}"
            )
        self.robot_name = robot_name_list[self.robot_id]
        # Single server per robot (MERGE_LEFT_RIGHT §1–§8).
        self.ik_server = ik_dict[self.robot_name]

        if self.obj_name is not None:
            self.planner_manager.update_obstacles(
                obstacle_avoidance_path_list=["dynamic"],
                env_ids=[self.env_id],
                obstacle_ignore_path_list=[self.obj_name],
            )

    def _get_grasp_candidates(self, grasp_dict: dict) -> list:
        """Get candidate list from functional_grasp/grasp dict (mirrors Grasp.py)."""
        if not grasp_dict or not isinstance(grasp_dict, dict):
            return []
        func_dict = grasp_dict.get("functional_grasp", {})
        grasp_only = grasp_dict.get("grasp", {})

        if self.functional_grasp is True:
            out = _extract_candidates_from_parts(func_dict, self.part)
            if not out:
                out = _extract_candidates_from_parts(grasp_only, self.part)
        else:
            out = _extract_candidates_from_parts(grasp_only, self.part)
            if not out:
                out = _extract_candidates_from_parts(func_dict, self.part)
        return out

    def _extract_coarse_poses_from_candidates(
        self, candidates: list
    ) -> torch.Tensor | None:
        """Extract single-arm coarse_grasp poses from candidates as ``(G, 7)``."""
        poses = []
        for c in candidates:
            coarse = c.get("coarse_grasp")
            if coarse is None:
                continue
            pos = _to_tensor(coarse["position"])
            ori = _to_tensor(coarse["orientation"])
            poses.append(torch.cat([pos, ori], dim=0))
        if not poses:
            return None
        return torch.stack(poses, dim=0)

    def _normalize_hand_joints(
        self, joints: torch.Tensor | None
    ) -> torch.Tensor | None:
        if joints is None:
            return None
        j = joints.flatten()
        if j.numel() > self.hand_joint_dim:
            return j[: self.hand_joint_dim].contiguous()
        if j.numel() < self.hand_joint_dim:
            pad = torch.zeros(
                self.hand_joint_dim - j.numel(), dtype=j.dtype, device=j.device
            )
            return torch.cat([j, pad], dim=0)
        return j

    def _commit_chosen(self, chosen: dict) -> None:
        """Commit coarse/fine/final poses + joints from the selected IK candidate."""
        dev = self.env.device

        def _t(k, key):
            return _to_tensor(k[key], device=dev)

        coarse = chosen["coarse_grasp"]
        fine = chosen.get("fine_grasp") or chosen["final_grasp"]
        final = chosen["final_grasp"]
        self.coarse_pose = torch.cat(
            [_t(coarse, "position"), _t(coarse, "orientation")], dim=0
        )
        self.fine_pose = torch.cat(
            [_t(fine, "position"), _t(fine, "orientation")], dim=0
        )
        self.final_pose = torch.cat(
            [_t(final, "position"), _t(final, "orientation")], dim=0
        )
        self.coarse_joints = self._normalize_hand_joints(_t(coarse, "joints"))
        self.fine_joints = self._normalize_hand_joints(_t(fine, "joints"))
        self.final_joints = self._normalize_hand_joints(_t(final, "joints"))

    def _start_job(self, pose_type: str, grasp_pose_tensor: torch.Tensor):
        self._grasp_token += 1
        token = self._grasp_token
        grasp_pose_list = grasp_pose_tensor.to(device=self.env.device)
        if grasp_pose_list.ndim != 2 or grasp_pose_list.shape[-1] != 7:
            raise ValueError(
                f"Expected grasp_pose_list [G,7], got {tuple(grasp_pose_list.shape)}"
            )
        self._grasp_job = {
            "token": token,
            "pose_type": pose_type,
            "pose_list": grasp_pose_list,
            "future": self._submit_goalset(grasp_pose_list),
        }

    def get_grasp_pose(self):
        """
        Async grasp pose selection. IK only on coarse_grasp poses.
        On success: commits coarse/final poses and joints from selected candidate.
        """
        if (
            self.coarse_pose is not None
            and self.pre_grasp_pose is not None
            and self.retrieval_pose is not None
        ):
            return self.coarse_pose

        self.current_state = "computing"
        self.current_action = None

        if self._grasp_job is None:
            if not hasattr(self.env, "get_grasp_pose"):
                self.current_state = "failed: no grasp pose method available"
                self._grasp_job = None
                return None
            grasp_list = self.env.get_grasp_pose(
                env_ids=[self.env_id],
                obj_name=self.obj_name,
                hand_type=self.hand_type,
                obj_id=self.obj_id,
            )
            missing_msg = f"failed: no {self.hand_type} grasp annotation"
            if not grasp_list or grasp_list[0] is None:
                self.current_state = missing_msg
                self._grasp_job = None
                return None

            candidates = self._get_grasp_candidates(grasp_list[0])
            if not candidates:
                self.current_state = "failed: no grasp candidates"
                self._grasp_job = None
                return None

            coarse_poses = self._extract_coarse_poses_from_candidates(candidates)
            if coarse_poses is None or coarse_poses.shape[0] == 0:
                self.current_state = "failed: no coarse_grasp poses"
                self._grasp_job = None
                return None

            self._grasp_token += 1
            self._grasp_job = {
                "token": self._grasp_token,
                "candidates": candidates,
                "pose_list": coarse_poses,
                "future": self._submit_goalset(coarse_poses),
            }
            return None

        job = self._grasp_job
        if job.get("token") != self._grasp_token:
            self._grasp_job = None
            return None

        fut = job.get("future")
        if fut is None or not fut.done():
            return None

        try:
            success_list, goalset_index_list, returned_env_ids = fut.result()
        except Exception as ex:
            self.current_state = f"failed: ik exception {ex}"
            self._grasp_job = None
            return None

        # Debug: log IK result
        if self.debug:
            print(
                f"DexGrasp env_id={self.env_id} goalset result: "
                f"success_list={success_list}, goalset_index_list={goalset_index_list}, "
                f"returned_env_ids={returned_env_ids}"
            )

        if not returned_env_ids or int(returned_env_ids[0]) != int(self.env_id):
            self.current_state = "failed: ik env mismatch"
            self._grasp_job = None
            return None

        selected_idx = -1
        if goalset_index_list is not None and len(goalset_index_list) >= 1:
            selected_idx = int(goalset_index_list[0])

        if selected_idx < 0 or not (len(success_list) >= 1 and bool(success_list[0])):
            self.current_state = "failed: ik found no solution"
            if self.debug:
                print(
                    f"DexGrasp env_id={self.env_id} ik failed: selected_idx={selected_idx}, "
                    f"success_list={success_list}, goalset_index_list={goalset_index_list}"
                )
            self._grasp_job = None
            return None

        candidates = job["candidates"]
        if selected_idx >= len(candidates):
            self.current_state = "failed: selected_idx out of bounds"
            self._grasp_job = None
            return None

        self._selected_grasp_idx = selected_idx
        chosen = candidates[selected_idx]

        self._commit_chosen(chosen)
        self.pre_grasp_pose = self._compute_pose_along_grasp_direction(
            self.coarse_pose, self.pre_grasp_offset, backward=True
        )
        self.retrieval_pose = self._compute_pose_upward(
            self.final_pose, self.retrieval_offset
        )
        min_z = float(self.pre_grasp_pose[2].item())
        self._pregrasp_use_retract = self.mobile and (
            min_z < self.pre_grasp_z_threshold
        )

        self.current_state = "ready"
        self._grasp_job = None
        return self.coarse_pose

    def _update_poses_from_object_pose(self) -> bool:
        """
        Update coarse/final/pre_grasp/retrieval poses from latest object pose
        using stored grasp pose id. Used when reactive=True during coarse/final grasp phases.
        """
        if self._selected_grasp_idx < 0:
            return False

        if not hasattr(self.env, "get_grasp_pose_updated"):
            return False
        result = self.env.get_grasp_pose_updated(
            env_ids=[self.env_id],
            obj_name=self.obj_name,
            obj_id=self.obj_id,
            obj_type=self.obj_type,
            hand_type=self.hand_type,
            selected_idx=self._selected_grasp_idx,
            functional_grasp=self.functional_grasp,
            part=self.part,
        )
        if result is None:
            return False
        self.coarse_pose = result["coarse_pose"].to(self.env.device)
        if result.get("fine_pose") is not None:
            self.fine_pose = result["fine_pose"].to(self.env.device)
        if result.get("fine_joints") is not None:
            self.fine_joints = result["fine_joints"].to(self.env.device)
        if result.get("final_pose") is not None:
            self.final_pose = result["final_pose"].to(self.env.device)
        if result.get("final_joints") is not None:
            self.final_joints = result["final_joints"].to(self.env.device)
        self.pre_grasp_pose = self._compute_pose_along_grasp_direction(
            self.coarse_pose, self.pre_grasp_offset, backward=True
        )
        self.retrieval_pose = self._compute_pose_upward(
            self.final_pose, self.retrieval_offset
        )
        if self.mobile:
            self._pregrasp_use_retract = (
                float(self.pre_grasp_pose[2].item()) < self.pre_grasp_z_threshold
            )
        return True

    def _get_env_origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[self.env_id].clone()

    def _grasp_pose_world_for_viz(self, pose: torch.Tensor) -> torch.Tensor:
        """Grasp targets are env-local; Isaac debug draw uses world frame (same as waypoint viz)."""
        device = self.env.device
        pose = pose.to(device=device)
        origin = self._get_env_origin().to(device=device)
        if origin.ndim > 1:
            origin = origin[0]
        out = pose.clone()
        out[:3] = pose[:3] + origin[:3]
        return out

    def _viz_grasp_for_phase(self, phase: str | None) -> None:
        """Current phase → visualize that phase's 7D grasp frame (single pose; replaces prior draw)."""
        if not self.viz_grasp or phase is None:
            return
        pose = None
        if phase == "pre_grasp":
            pose = self.pre_grasp_pose
        elif phase == "coarse_grasp":
            pose = self.coarse_pose
        elif phase == "fine_grasp":
            pose = self.fine_pose
        elif phase == "final_grasp":
            pose = self.final_pose
        elif phase == "retrieval":
            pose = self.retrieval_pose
        if pose is not None:
            dev = self.env.device
            pose = pose.to(dev)
            visualize_grasp_pose([self._grasp_pose_world_for_viz(pose)])

    def _compute_pose_along_grasp_direction(
        self, grasp_pose, offset_distance, backward=True
    ):
        """Pre-grasp offset along one axis of the grasp frame. Axis 0=x, 1=y, 2=z."""
        device = self.env.device
        grasp_pose = _to_tensor(grasp_pose, device=device)
        grasp_pos = grasp_pose[:3]
        grasp_quat = grasp_pose[3:7]
        rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))
        approach_axis = 1
        grasp_direction = rot_matrix[0, :, approach_axis]
        grasp_direction = grasp_direction / torch.norm(grasp_direction)
        offset = grasp_direction * offset_distance
        new_pos = grasp_pos - offset if backward else grasp_pos + offset
        return torch.cat([new_pos, grasp_quat], dim=0)

    def _compute_pose_upward(self, grasp_pose, offset_distance):
        device = self.env.device
        grasp_pose = _to_tensor(grasp_pose, device=device)
        grasp_pos = grasp_pose[:3].clone()
        grasp_pos[2] += offset_distance
        return torch.cat([grasp_pos, grasp_pose[3:7]], dim=0)

    def refresh(self, action: list[Any]):
        if len(action) >= 8:
            new_command = [
                "DexGrasp",
                int(action[1]),
                int(action[2]),
                action[3],
                action[4],
                action[5],
                action[6],
                action[7],
            ]
        else:
            new_func = (
                action[4]
                if len(action) >= 6
                else getattr(self.config, "functional_grasp", True)
            )
            new_part = (
                action[5]
                if len(action) >= 6
                else getattr(self.config, "functional_part", None)
            )
            new_command = [
                "DexGrasp",
                int(getattr(self.config, "robot_id", 0)),
                int(getattr(self.config, "hand_id", 0)),
                action[1],
                action[2],
                action[3],
                new_func,
                new_part,
            ]
        if self.current_command != new_command or self.current_phase is None:
            self.reset(action)

    def step(self):
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        # Placeholder during init (obj not set) or computing (grasp pose not ready)
        if self.obj_type is None or self.obj_name is None or self.obj_id is None:
            self.current_action = self._build_placeholder_action()
            return self.current_action

        if (
            self.coarse_pose is None
            or self.pre_grasp_pose is None
            or self.retrieval_pose is None
        ):
            self.get_grasp_pose()
            if self.current_state == "computing":
                self.current_action = self._build_placeholder_action()
                return self.current_action
            if (
                self.coarse_pose is None
                or self.pre_grasp_pose is None
                or self.retrieval_pose is None
            ):
                self.current_state = "failed"
                self.current_action = "Failed"
                return "Failed"

        self.current_state = "running"
        dev = self.env.device

        if self.viz_grasp and self.current_phase != self._last_viz_phase:
            self._viz_grasp_for_phase(self.current_phase)
            self._last_viz_phase = self.current_phase

        hand_zeros = torch.zeros(self.hand_joint_dim, dtype=torch.float32, device=dev)

        if self.current_phase == "pre_grasp":
            arm_pose = self.pre_grasp_pose.to(dev)
            self.current_action = self._build_motion_action(
                arm_pose, hand_zeros, "pre_grasp"
            )
            return self.current_action

        elif self.current_phase == "coarse_grasp":
            if self.reactive and not self._coarse_grasp_pose_updated:
                self._update_poses_from_object_pose()
                self._coarse_grasp_pose_updated = True
            arm_pose = self.coarse_pose.to(dev)
            self.current_action = self._build_motion_action(
                arm_pose, hand_zeros, "coarse_grasp"
            )
            return self.current_action

        elif self.current_phase == "fine_grasp":
            if self.reactive and not self._fine_grasp_pose_updated:
                self._update_poses_from_object_pose()
                self._fine_grasp_pose_updated = True
            arm_pose = self.final_pose.to(dev)
            self.current_action = self._build_motion_action(
                arm_pose, hand_zeros, "fine_grasp"
            )
            return self.current_action

        elif self.current_phase == "final_grasp":
            if self.reactive and not self._final_grasp_pose_updated:
                self._update_poses_from_object_pose()
                self._final_grasp_pose_updated = True
            arm_pose = self.final_pose.to(dev)
            hand_joints = self.final_joints.to(dev)
            self.current_action = self._build_motion_action(
                arm_pose, hand_joints, "final_grasp"
            )
            return self.current_action

        elif self.current_phase == "retrieval":
            arm_pose = self.retrieval_pose.to(dev)
            hand_joints = self.final_joints.to(dev)
            self.current_action = self._build_motion_action(
                arm_pose, hand_joints, "retrieval"
            )
            return self.current_action

        else:
            self.current_state = "failed"
            self.current_action = None
            return None

    def update(self, info: dict) -> dict:
        base = {
            "atomic_skill_type": "DexGrasp",
            "command": self.current_command,
            "action": self.current_action,
            "phase": self.current_phase,
        }

        if self.current_state == "computing":
            return {**base, "finished": False, "state": "computing", "truncated": 0}

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
        # Placeholder refresh / planner swap (GlobalPlanner reports truncated=5, no log line)
        if trunc == 5:
            return {
                **base,
                "finished": False,
                "state": self.current_state or "running",
                "truncated": 0,
            }

        if env_gp.get("finished", False):
            if self.current_phase == "pre_grasp":
                self.current_phase = "coarse_grasp"
                self._coarse_grasp_pose_updated = (
                    False  # Allow one update when entering coarse_grasp
                )
                if self.debug:
                    print(f"[DexGrasp] env_id={self.env_id} phase=coarse_grasp")
                return {
                    **base,
                    "finished": False,
                    "state": "running: coarse grasp",
                    "truncated": 0,
                    "phase": "coarse_grasp",
                }
            elif self.current_phase == "coarse_grasp":
                self.current_phase = "fine_grasp"
                self._fine_grasp_pose_updated = (
                    False  # Allow one update when entering fine_grasp
                )
                if self.debug:
                    print(f"[DexGrasp] env_id={self.env_id} phase=fine_grasp")
                return {
                    **base,
                    "finished": False,
                    "state": "running: fine grasp",
                    "truncated": 0,
                    "phase": "fine_grasp",
                }
            elif self.current_phase == "fine_grasp":
                self.current_phase = "final_grasp"
                self._final_grasp_pose_updated = False
                if self.debug:
                    print(f"[DexGrasp] env_id={self.env_id} phase=final_grasp")
                return {
                    **base,
                    "finished": False,
                    "state": "running: closing hand",
                    "truncated": 0,
                    "phase": "final_grasp",
                }
            elif self.current_phase == "final_grasp":
                self.current_phase = "retrieval"
                if self.debug:
                    print(f"[DexGrasp] env_id={self.env_id} phase=retrieval")
                return {
                    **base,
                    "finished": False,
                    "state": "running: retrieval",
                    "truncated": 0,
                    "phase": "retrieval",
                }
            elif self.current_phase == "retrieval":
                self.current_state = "finished"
                if self.debug:
                    print(f"[DexGrasp] env_id={self.env_id} phase=completed")
                return {
                    **base,
                    "finished": True,
                    "state": "finished",
                    "truncated": 0,
                    "phase": "completed",
                }

        return {**base, "finished": False, "state": "running", "truncated": 0}
