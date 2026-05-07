"""
BiDexGrasp atomic skill for paired bimanual dexterous hand grasping.

Bimanual sibling of :class:`DexGrasp` — drives BOTH dexterous hands
synchronously through every grasp phase using paired (dual) IK so a single
goalset index satisfies right + left arms simultaneously.

Phases (both arms move in lock-step):
  1. pre_grasp    — both arms backed off from coarse_grasp; hands open
  2. coarse_grasp — both arms to coarse pose; hands open
  3. fine_grasp   — both arms to final pose; hands open (precise positioning)
  4. final_grasp  — both hands close to final joints
  5. retrieval    — both arms lift z-up; hands keep final joints

The paired grasp annotation is loaded by the env via
``get_bimanual_grasp_pose`` (see :class:`BiDexGraspEnv` /
:class:`LocoBiGraspEnv`). Each candidate carries::

    {"left_hand": {coarse, fine, final}, "right_hand": {coarse, fine, final}}

IK runs once on the bimanual ``coarse_grasp`` goalset (right_7 + left_7)
and the chosen index is reused for the rest of the phases.
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


def _to_tensor(x, dtype=torch.float32, device=None):
    if isinstance(x, torch.Tensor):
        out = x.detach().clone().to(dtype=dtype)
        return out.to(device) if device is not None else out
    return torch.tensor(x, dtype=dtype, device=device)


def _extract_candidates_from_parts(parts: dict, part_name: str | None) -> list:
    out = []
    if part_name and part_name in parts and isinstance(parts[part_name], list):
        out.extend(parts[part_name])
    if not out:
        for v in parts.values():
            if isinstance(v, list):
                out.extend(v)
    return out


class BiDexGrasp(AtomicSkill):
    """Bimanual dexterous grasp — paired IK on coarse_grasp goalset."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_phase = None
        self.coarse_pose = None
        self.fine_pose = None
        self.final_pose = None
        self.pre_grasp_pose = None
        self.retrieval_pose = None
        self.coarse_joints = None
        self.fine_joints = None
        self.final_joints = None
        self.functional_grasp = True
        self.part = None

        self.pre_grasp_offset = float(getattr(config, "pre_grasp_offset", 0.05))
        self.retrieval_offset = float(getattr(config, "retrieval_offset", 0.2))
        self.mobile = bool(getattr(config, "mobile", False))
        self.hand_type = getattr(config, "hand_type", "sharpa")
        # Per-hand joint dim. sharpa=22, dex3_1=7, xhand=12.
        hand_joint_dim_map = getattr(
            config, "hand_joint_dim_map", {"sharpa": 22, "xhand": 12, "dex3_1": 7}
        )
        self.hand_joint_dim = int(
            getattr(
                config, "hand_joint_dim", hand_joint_dim_map.get(self.hand_type, 22)
            )
        )
        self.pre_grasp_z_threshold = float(
            getattr(config, "pre_grasp_z_threshold", 0.4)
        )
        self._pregrasp_use_retract = False
        self._placeholder_gp_key = "MobileMoveL" if self.mobile else "MoveL"
        self._placeholder_mode = 3 if self.mobile else -1
        self.debug = bool(getattr(config, "debug", False))
        self.viz_grasp = bool(getattr(config, "viz_grasp", True))
        self.reactive = bool(getattr(config, "reactive", True))

        self._grasp_job: dict | None = None
        self._grasp_token: int = 0
        self._selected_grasp_idx: int = -1
        self._coarse_grasp_pose_updated = False
        self._fine_grasp_pose_updated = False
        self._final_grasp_pose_updated = False
        self._last_viz_phase = None

        self.robot_id = 0
        # Always paired: hand_id is fixed at -1 for the bimanual branch in
        # downstream IK / planner clients (mirrors :class:`BiGrasp`).
        self.hand_id = -1
        self.robot_name: str | None = None
        self.ik_server = None
        self.planner_manager: PlannerManager | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _arm_dim(self) -> int:
        return 14

    def _hand_action_dim(self) -> int:
        return 2 * self.hand_joint_dim

    def _build_placeholder_action(self) -> dict:
        dev = self.env.device
        eef = torch.full(
            (self._arm_dim() + self._hand_action_dim(),),
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
        if phase in ("pre_grasp", "coarse_grasp"):
            return -1
        return 0

    def _mobile_mode_for_phase(self, phase: str) -> int:
        # MobileMoveL planner_mode header (see MobileMoveL.py docstring):
        #   3  = no IK, FREE base, always MotionGen
        #  -1  = submit IK, base mode from IK, MotionGen if dist > threshold
        #   0  = NO IK, LOCKED base, NO MotionGen → single-step arm target
        #         straight to Pink IK (cheapest path).
        if phase == "pre_grasp":
            return 3
        if phase == "coarse_grasp":
            return -1
        if phase in ("fine_grasp", "final_grasp"):
            # Mode 0: by this point the wrists are already at the coarse_grasp
            # pose; fine_grasp + final_grasp are sub-cm refinements where
            # re-running IK / MotionGen costs more than it helps and the base
            # MUST stay locked (any base drift here would slip the paired
            # grasp targets relative to the bin). Pink IK alone closes the
            # last few mm to the final_pose targets while the hands close.
            return 0
        if phase == "retrieval":
            # Mode 0 again: hands are gripped on the bin, just need a
            # straight-up Pink IK lift via the arms. Locked base prevents
            # the wheels from drifting while the bin is held.
            return 0
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
                    (self.robot_id, self.hand_id, self._move_l_mode_for_phase(phase)),
                    eef_action,
                )
            }
        key = self._mobile_planner_key_for_phase(phase)
        return {
            key: (
                (self.robot_id, self.hand_id, self._mobile_mode_for_phase(phase)),
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

    def _submit_paired_goalset(self, grasp_pose_list: torch.Tensor):
        """``grasp_pose_list`` is ``(G, 14)`` = ``[right_7, left_7]`` per row.

        Both slots are real → paired IK picks one ``g`` satisfying both arms.
        """
        if grasp_pose_list.ndim != 2 or grasp_pose_list.shape[-1] != 14:
            raise ValueError(
                f"BiDexGrasp _submit_paired_goalset expects (G, 14); "
                f"got {tuple(grasp_pose_list.shape)}"
            )
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        target = grasp_pose_list.to(device=self.env.device).unsqueeze(0).contiguous()
        is_dual_ik = bool(getattr(self.ik_server, "dual_mode", False))
        if is_dual_ik:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=robot_states_dict,
                mode="goalset",
                lock_base=False,
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=robot_states_dict,
                mode="goalset",
            )
        return self.ik_server.submit_ik(req)

    # ------------------------------------------------------------------
    # Reset / refresh
    # ------------------------------------------------------------------

    def reset(self, action: list[Any]):
        # action: [BiDexGrasp, robot_id, obj_type, obj_name, obj_id, optional functional_grasp, optional part]
        if len(action) >= 7:
            self.robot_id = int(action[1])
            self.obj_type = action[2]
            self.obj_name = action[3]
            self.obj_id = action[4]
            self.functional_grasp = action[5]
            self.part = action[6]
        else:
            self.robot_id = int(getattr(self.config, "robot_id", 0))
            self.obj_type = action[1]
            self.obj_name = action[2]
            self.obj_id = action[3]
            if len(action) >= 5:
                self.functional_grasp = action[4]
                self.part = action[5] if len(action) >= 6 else None
            else:
                self.functional_grasp = getattr(self.config, "functional_grasp", True)
                self.part = getattr(self.config, "functional_part", None)

        self.hand_id = -1  # always paired
        self.current_state = "ready"
        self.current_command = [
            "BiDexGrasp",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.functional_grasp,
            self.part,
        ]
        self.current_phase = "pre_grasp"
        if self.debug:
            print(f"[BiDexGrasp] env_id={self.env_id} phase=pre_grasp (reset)")

        self.coarse_pose = None
        self.fine_pose = None
        self.final_pose = None
        self.pre_grasp_pose = None
        self.retrieval_pose = None
        self.coarse_joints = None
        self.fine_joints = None
        self.final_joints = None
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
                f"BiDexGrasp: robot_id {self.robot_id} out of range for "
                f"robot_name_list={robot_name_list}"
            )
        self.robot_name = robot_name_list[self.robot_id]
        self.ik_server = ik_dict[self.robot_name]

        if self.obj_name is not None:
            self.planner_manager.update_obstacles(
                obstacle_avoidance_path_list=["dynamic"],
                env_ids=[self.env_id],
                obstacle_ignore_path_list=[self.obj_name],
            )

    def refresh(self, action: list[Any]):
        if len(action) >= 7:
            new_command = [
                "BiDexGrasp",
                int(action[1]),
                action[2],
                action[3],
                action[4],
                action[5],
                action[6],
            ]
        else:
            new_func = (
                action[4]
                if len(action) >= 5
                else getattr(self.config, "functional_grasp", True)
            )
            new_part = (
                action[5]
                if len(action) >= 6
                else getattr(self.config, "functional_part", None)
            )
            new_command = [
                "BiDexGrasp",
                int(getattr(self.config, "robot_id", 0)),
                action[1],
                action[2],
                action[3],
                new_func,
                new_part,
            ]
        if self.current_command != new_command or self.current_phase is None:
            self.reset(action)

    # ------------------------------------------------------------------
    # Grasp candidate extraction (paired)
    # ------------------------------------------------------------------

    def _get_grasp_candidates(self, grasp_dict: dict) -> list:
        """Pick paired-candidate list from functional_grasp/grasp dict."""
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
        """Build ``(G, 14)`` = ``[right_7, left_7]`` coarse goalset."""
        rows = []
        for c in candidates:
            if not (isinstance(c, dict) and "left_hand" in c and "right_hand" in c):
                continue
            r = c["right_hand"].get("coarse_grasp")
            left_cg = c["left_hand"].get("coarse_grasp")
            if r is None or left_cg is None:
                continue
            r_pose = torch.cat(
                [
                    _to_tensor(r["position"]).flatten()[:3],
                    _to_tensor(r["orientation"]).flatten()[:4],
                ],
                dim=0,
            )
            l_pose = torch.cat(
                [
                    _to_tensor(left_cg["position"]).flatten()[:3],
                    _to_tensor(left_cg["orientation"]).flatten()[:4],
                ],
                dim=0,
            )
            rows.append(torch.cat([r_pose, l_pose], dim=0))
        if not rows:
            return None
        return torch.stack(rows, dim=0)

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

    def _pose7_from_phase(self, phase: dict) -> torch.Tensor:
        dev = self.env.device
        pos = _to_tensor(phase["position"], device=dev).flatten()[:3]
        ori = _to_tensor(phase["orientation"], device=dev).flatten()[:4]
        return torch.cat([pos, ori], dim=0)

    def _commit_chosen(self, chosen: dict) -> None:
        """Pack ``[right_7, left_7]`` poses and ``[right_joints, left_joints]`` per phase."""
        dev = self.env.device
        r = chosen["right_hand"]
        left_hand = chosen["left_hand"]
        r_coarse = self._pose7_from_phase(r["coarse_grasp"])
        l_coarse = self._pose7_from_phase(left_hand["coarse_grasp"])
        r_fine_src = r.get("fine_grasp") or r["final_grasp"]
        l_fine_src = left_hand.get("fine_grasp") or left_hand["final_grasp"]
        r_fine = self._pose7_from_phase(r_fine_src)
        l_fine = self._pose7_from_phase(l_fine_src)
        r_final = self._pose7_from_phase(r["final_grasp"])
        l_final = self._pose7_from_phase(left_hand["final_grasp"])

        self.coarse_pose = torch.cat([r_coarse, l_coarse], dim=0).to(dev)
        self.fine_pose = torch.cat([r_fine, l_fine], dim=0).to(dev)
        self.final_pose = torch.cat([r_final, l_final], dim=0).to(dev)

        def _side_joints(side: dict, key: str) -> torch.Tensor:
            src = side.get(key)
            if src is None or "joints" not in src:
                return torch.zeros(self.hand_joint_dim, dtype=torch.float32, device=dev)
            return self._normalize_hand_joints(_to_tensor(src["joints"], device=dev))

        self.coarse_joints = torch.cat(
            [
                _side_joints(r, "coarse_grasp"),
                _side_joints(left_hand, "coarse_grasp"),
            ],
            dim=0,
        )
        self.fine_joints = torch.cat(
            [
                _side_joints(r, "fine_grasp")
                if r.get("fine_grasp") is not None
                else _side_joints(r, "final_grasp"),
                _side_joints(left_hand, "fine_grasp")
                if left_hand.get("fine_grasp") is not None
                else _side_joints(left_hand, "final_grasp"),
            ],
            dim=0,
        )
        self.final_joints = torch.cat(
            [
                _side_joints(r, "final_grasp"),
                _side_joints(left_hand, "final_grasp"),
            ],
            dim=0,
        )

    # ------------------------------------------------------------------
    # IK orchestration
    # ------------------------------------------------------------------

    def get_grasp_pose(self):
        """Async paired-grasp selection on the coarse_grasp goalset."""
        if (
            self.coarse_pose is not None
            and self.pre_grasp_pose is not None
            and self.retrieval_pose is not None
        ):
            return self.coarse_pose

        self.current_state = "computing"
        self.current_action = None

        if self._grasp_job is None:
            if not hasattr(self.env, "get_bimanual_grasp_pose"):
                self.current_state = "failed: env lacks get_bimanual_grasp_pose"
                self._grasp_job = None
                return None
            grasp_list = self.env.get_bimanual_grasp_pose(
                env_ids=[self.env_id],
                obj_name=self.obj_name,
                hand_type=self.hand_type,
                obj_id=self.obj_id,
            )
            if not grasp_list or grasp_list[0] is None:
                self.current_state = f"failed: no {self.hand_type}_bimanual annotation"
                self._grasp_job = None
                return None

            candidates = self._get_grasp_candidates(grasp_list[0])
            if not candidates:
                self.current_state = "failed: no paired grasp candidates"
                self._grasp_job = None
                return None

            coarse_poses = self._extract_coarse_poses_from_candidates(candidates)
            if coarse_poses is None or coarse_poses.shape[0] == 0:
                self.current_state = "failed: no paired coarse_grasp poses"
                self._grasp_job = None
                return None

            self._grasp_token += 1
            self._grasp_job = {
                "token": self._grasp_token,
                "candidates": candidates,
                "pose_list": coarse_poses,
                "future": self._submit_paired_goalset(coarse_poses),
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

        if self.debug:
            print(
                f"BiDexGrasp env_id={self.env_id} goalset result: "
                f"success={success_list}, idx={goalset_index_list}, "
                f"envs={returned_env_ids}"
            )

        if not returned_env_ids or int(returned_env_ids[0]) != int(self.env_id):
            self.current_state = "failed: ik env mismatch"
            self._grasp_job = None
            return None

        selected_idx = -1
        if goalset_index_list is not None and len(goalset_index_list) >= 1:
            selected_idx = int(goalset_index_list[0])
        if selected_idx < 0 or not (len(success_list) >= 1 and bool(success_list[0])):
            self.current_state = "failed: paired ik no solution"
            self._grasp_job = None
            return None

        candidates = job["candidates"]
        if selected_idx >= len(candidates):
            self.current_state = "failed: selected_idx out of bounds"
            self._grasp_job = None
            return None

        self._selected_grasp_idx = selected_idx
        self._commit_chosen(candidates[selected_idx])
        self.pre_grasp_pose = self._compute_pose_along_grasp_direction(
            self.coarse_pose, self.pre_grasp_offset, backward=True
        )
        self.retrieval_pose = self._compute_pose_upward(
            self.final_pose, self.retrieval_offset
        )
        # min z across both arms' pre-grasp slots (right z @ idx 2, left z @ idx 9)
        min_z = float(min(self.pre_grasp_pose[2].item(), self.pre_grasp_pose[9].item()))
        self._pregrasp_use_retract = self.mobile and (
            min_z < self.pre_grasp_z_threshold
        )

        self.current_state = "ready"
        self._grasp_job = None
        return self.coarse_pose

    def _update_poses_from_object_pose(self) -> bool:
        """Reactive refresh: re-read paired candidate at the latest object pose."""
        if self._selected_grasp_idx < 0:
            return False
        if not hasattr(self.env, "get_bimanual_grasp_pose_updated"):
            return False
        chosen = self.env.get_bimanual_grasp_pose_updated(
            env_ids=[self.env_id],
            obj_name=self.obj_name,
            obj_id=self.obj_id,
            obj_type=self.obj_type,
            hand_type=self.hand_type,
            selected_idx=self._selected_grasp_idx,
            functional_grasp=self.functional_grasp,
            part=self.part,
        )
        if chosen is None:
            return False
        self._commit_chosen(chosen)
        self.pre_grasp_pose = self._compute_pose_along_grasp_direction(
            self.coarse_pose, self.pre_grasp_offset, backward=True
        )
        self.retrieval_pose = self._compute_pose_upward(
            self.final_pose, self.retrieval_offset
        )
        if self.mobile:
            min_z = float(
                min(self.pre_grasp_pose[2].item(), self.pre_grasp_pose[9].item())
            )
            self._pregrasp_use_retract = min_z < self.pre_grasp_z_threshold
        return True

    # ------------------------------------------------------------------
    # Pose math + viz
    # ------------------------------------------------------------------

    def _get_env_origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[self.env_id].clone()

    def _grasp_pose_world_for_viz(self, pose: torch.Tensor) -> torch.Tensor:
        device = self.env.device
        pose = pose.to(device=device)
        origin = self._get_env_origin().to(device=device)
        if origin.ndim > 1:
            origin = origin[0]
        out = pose.clone()
        out[:3] = pose[:3] + origin[:3]
        return out

    def _viz_grasp_for_phase(self, phase: str | None) -> None:
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
        if pose is None:
            return
        dev = self.env.device
        pose = pose.to(dev)
        poses_world = [
            self._grasp_pose_world_for_viz(pose[:7]),
            self._grasp_pose_world_for_viz(pose[7:]),
        ]
        visualize_grasp_pose(poses_world)

    def _compute_pose_along_grasp_direction(
        self, grasp_pose, offset_distance, backward=True
    ):
        """Pregrasp standoff for sharpa-style "hands facing each other" rigs.

        Sharpa's paired annotation does NOT encode a clean per-hand approach
        axis in the local rotation frame: both R and L hand quats land their
        local +Y on the SAME bin-y direction (not each side's outward radial),
        so backing off along ``rot_matrix[:, 1]`` collapses both wrists toward
        the bin centerline instead of separating them.

        Geometrically the only stable approach direction is the **inter-hand
        line** — each hand backs off along the unit vector pointing FROM the
        partner hand TO itself (away from the partner). In the parked vega
        scene with bin yaw=90°, this line projects onto world Y, which is
        the user's "y 轴" intuition. Robust to bin yaw because we derive the
        direction from the pair's xyz, not from any single quat.

        Bimanual input ``[right_7, left_7]``: split, compute the inter-hand
        vector (xyz only; orientation untouched), offset each side, repack.

        ``backward=True`` (caller convention) ⇒ pregrasp = grasp + outward
        offset. The "backward" flag is consumed at the pair level: True
        means each hand moves AWAY from its partner (the standard pregrasp
        direction); False would move them TOWARD each other (reserved for
        post-grasp converge scenarios).
        """
        device = self.env.device
        grasp_pose = _to_tensor(grasp_pose, device=device)
        if grasp_pose.numel() != 14:
            # Single-arm fallback (shouldn't be hit in BiDexGrasp; kept for
            # API symmetry with DexGrasp callers).
            return grasp_pose.clone()

        r_pose = grasp_pose[:7]
        l_pose = grasp_pose[7:]
        r_pos = r_pose[:3]
        l_pos = l_pose[:3]

        inter = r_pos - l_pos
        norm = torch.norm(inter)
        if float(norm) < 1e-6:
            # Hands annotated at identical xyz — degenerate pair. Skip the
            # offset entirely rather than divide by zero; the IK should
            # reject this candidate downstream.
            return grasp_pose.clone()
        unit = inter / norm

        sign = 1.0 if backward else -1.0
        r_offset = sign * offset_distance * unit
        l_offset = -sign * offset_distance * unit

        r_new_pos = r_pos + r_offset
        l_new_pos = l_pos + l_offset

        r_new = torch.cat([r_new_pos, r_pose[3:7]], dim=0)
        l_new = torch.cat([l_new_pos, l_pose[3:7]], dim=0)
        return torch.cat([r_new, l_new], dim=0)

    def _compute_pose_upward(self, grasp_pose, offset_distance):
        device = self.env.device
        grasp_pose = _to_tensor(grasp_pose, device=device)
        if grasp_pose.numel() == 14:
            right = self._compute_pose_upward(grasp_pose[:7], offset_distance)
            left = self._compute_pose_upward(grasp_pose[7:], offset_distance)
            return torch.cat([right, left], dim=0)
        grasp_pos = grasp_pose[:3].clone()
        grasp_pos[2] += offset_distance
        return torch.cat([grasp_pos, grasp_pose[3:7]], dim=0)

    # ------------------------------------------------------------------
    # Step / update
    # ------------------------------------------------------------------

    def step(self):
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

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

        hand_zeros = torch.zeros(
            self._hand_action_dim(), dtype=torch.float32, device=dev
        )

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
            "atomic_skill_type": "BiDexGrasp",
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
                self._coarse_grasp_pose_updated = False
                if self.debug:
                    print(f"[BiDexGrasp] env_id={self.env_id} phase=coarse_grasp")
                return {
                    **base,
                    "finished": False,
                    "state": "running: coarse grasp",
                    "truncated": 0,
                    "phase": "coarse_grasp",
                }
            elif self.current_phase == "coarse_grasp":
                self.current_phase = "fine_grasp"
                self._fine_grasp_pose_updated = False
                if self.debug:
                    print(f"[BiDexGrasp] env_id={self.env_id} phase=fine_grasp")
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
                    print(f"[BiDexGrasp] env_id={self.env_id} phase=final_grasp")
                return {
                    **base,
                    "finished": False,
                    "state": "running: closing hands",
                    "truncated": 0,
                    "phase": "final_grasp",
                }
            elif self.current_phase == "final_grasp":
                self.current_phase = "retrieval"
                if self.debug:
                    print(f"[BiDexGrasp] env_id={self.env_id} phase=retrieval")
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
                    print(f"[BiDexGrasp] env_id={self.env_id} phase=completed")
                return {
                    **base,
                    "finished": True,
                    "state": "finished",
                    "truncated": 0,
                    "phase": "completed",
                }

        return {**base, "finished": False, "state": "running", "truncated": 0}
