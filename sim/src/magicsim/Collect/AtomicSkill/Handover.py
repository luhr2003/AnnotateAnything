"""Handover atomic skill: bimanual handover on DualFranka.

Drives the full state machine described in
``Task/TableTop/Env/HandoverEnv.py``:

    Phase                    Active arm(s)            What runs
    ------------------------ ------------------------ ----------------------
    left_pre_grasp           left only  (hand_id=1)   MoveL → grasp - 0.15·z
    left_grasp               left only                MoveL → grasp pose
    left_close               left gripper             ParallelGripper close
    left_lift                left only                MoveL → grasp + 0.15·z_world
    compute_handover         (no MoveL)               PAIRED IK goalset
    right_pre_grasp          BOTH (hand_id=-1)        MoveL → (right_pre, left_handover)
    right_grasp              BOTH                     MoveL → (right_grasp, left_handover)
    right_close              right gripper            ParallelGripper close
    left_open                left gripper             ParallelGripper open
    left_retract             BOTH                     MoveL → (right_grasp, left_back-off)

Two distinct IK solves happen:

    * single-arm IK at ``reset`` (left only) — pick a reachable left grasp.
    * paired bimanual IK at ``compute_handover`` — for every
      (handover_mug_pose, right_local) pair, build (right_world, left_world)
      and submit ONE goalset; curobo argmins jointly so the chosen ``g``
      keeps left in its grasp AND lets right reach a valid grasp.

The paired solve is the critical step: by feeding the LEFT slot the
exact world pose the left wrist must occupy in order to keep gripping
the mug at the proposed handover orientation, the IK fails any
candidate where the left arm cannot stay there. No NaN-disable for the
left slot — that was the user constraint.
"""

from typing import Any, List

import concurrent.futures
import torch
from loguru import logger as log
from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Planner.PlannerManager import PlannerManager
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


_PHASES = [
    "left_pre_grasp",
    "left_grasp",
    "left_close",
    "left_lift",
    "compute_handover",
    "right_pre_grasp",
    "right_grasp",
    "right_close",
    "left_open",
    "left_retract",
]


class Handover(AtomicSkill):
    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = 0
        # Single-arm packing helper inside ``pack_single_arm_goalset`` reads
        # ``self.hand_id``; we flip it between solves (1 for left-only,
        # ignored for the paired build).
        self.hand_id = 1

        self.pre_grasp_offset = float(getattr(config, "pre_grasp_offset", 0.15))
        self.lift_offset = float(getattr(config, "lift_offset", 0.15))
        self.retract_offset = float(getattr(config, "retract_offset", 0.15))
        self.right_pre_grasp_offset = float(
            getattr(config, "right_pre_grasp_offset", 0.15)
        )
        # Hard-coded part split mirroring TestHandoverEnv.py open-loop:
        # left grabs body (sturdy multi-candidate pool), right grabs
        # handle (the functional grasp). ``None`` falls back to "all
        # parts merged" — the original behavior.
        self.left_part: str | None = getattr(config, "left_part", "body")
        self.right_part: str | None = getattr(config, "right_part", "handle")
        # Cap on right-arm grasp candidates fed into the paired handover
        # IK. Smaller-than-handle-pool means subsample; ≥pool size keeps
        # all (mug 8684 handle pool = 22).
        self.max_right_pool = int(getattr(config, "max_right_pool", 22))
        # Mug-pose generator knobs (mirror open-loop smoke params).
        self.n_yaws = int(getattr(config, "n_yaws", 8))
        self.pitch_degs = tuple(getattr(config, "pitch_degs", (-30.0, 0.0, 30.0)))
        self.roll_degs = tuple(getattr(config, "roll_degs", (-30.0, 0.0, 30.0)))
        self.viz = bool(getattr(config, "viz", False))

        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None

        self.current_phase: str | None = None
        self.current_command: list | None = None

        # Resolved at reset.
        self.robot_name: str | None = None
        self.ik_server = None
        self.planner_manager: PlannerManager | None = None

        # Pool of grasp candidates in MUG-LOCAL frame (Tensor[N, 7]).
        self._grasp_pool_local: torch.Tensor | None = None
        # Same pool transformed by the mug pose at IK-submit time
        # (Tensor[N, 7] in world frame). Used to solve the left grasp.
        self._grasp_pool_world: torch.Tensor | None = None

        # Phase 1 results (left grasp).
        self.left_grasp_local: torch.Tensor | None = None
        self.left_grasp_world: torch.Tensor | None = None
        self.left_pre_grasp_world: torch.Tensor | None = None
        self.left_lift_world: torch.Tensor | None = None

        # Phase 5 results (handover).
        self.left_handover_eef: torch.Tensor | None = None  # world frame, 7
        self.right_grasp_world: torch.Tensor | None = None  # world frame, 7
        self.right_pre_grasp_world: torch.Tensor | None = None
        self.left_retract_world: torch.Tensor | None = None
        # Saved at compute_handover so right phases can re-derive their
        # targets from the LATEST mug world pose (drift correction).
        self.chosen_right_local: torch.Tensor | None = None
        # Tracks which phase last triggered a right-target refresh, so
        # we only re-read mug pose at phase ENTRY (not every tick).
        self._refreshed_for_phase: str | None = None
        # Right-arm hold pose captured at reset (after env settle). All
        # left-only phases (pre/grasp/close/lift) emit hand_id=-1 16D
        # actions with right=right_hold to avoid the ik_dual_diff
        # NaN-fill frame bug: ``process_actions`` fills NaN rows with
        # current EE pose IN BASE FRAME, then the downstream
        # ``_transform_command_to_base`` re-applies the world→base
        # transform assuming world-frame input — drives right wrist
        # to a garbage target. Sending the right slot explicitly with
        # the world-frame current pose sidesteps that path entirely.
        self.right_hold: torch.Tensor | None = None

        # Async IK job tracking. One slot for left-grasp solve, one for
        # paired handover solve — they never overlap (sequential phases).
        self._ik_job: dict | None = None
        self._ik_token: int = 0

    # ==================================================================
    # Resolution helpers
    # ==================================================================

    def _get_planner_manager(self):
        pm = getattr(self.env.scene, "planner_manager", None)
        if pm is None:
            raise RuntimeError("PlannerManager not available.")
        return pm

    def _resolve_robot_name(self) -> str:
        if self.robot_name is not None:
            return self.robot_name
        rm = getattr(self.env.scene, "robot_manager", None)
        if rm is not None and isinstance(getattr(rm, "robots", None), dict):
            self.robot_name = next(iter(rm.robots.keys()))
            return self.robot_name
        raise RuntimeError("Unable to resolve robot_name.")

    def _get_robot_state(self) -> dict:
        states = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        if isinstance(states, dict):
            name = self._resolve_robot_name()
            return states.get(name, next(iter(states.values())))
        return states

    def _robot_state_dict(self) -> dict:
        rs = self._get_robot_state()
        return {
            "base_pos": rs["base_pos"],
            "base_quat": rs["base_quat"],
            "joint_pos": rs["joint_pos"],
            "joint_vel": rs["joint_vel"],
        }

    def _get_eef_pose_for_arm(self, hand_slot: int) -> torch.Tensor:
        """Return current eef pose 7-vec (world frame) for slot 0=right, 1=left."""
        rs = self._get_robot_state()
        eef_pos = rs["eef_pos"]
        eef_quat = rs["eef_quat"]
        if eef_pos.dim() == 3:
            pos = eef_pos[self.env_id, hand_slot]
            quat = eef_quat[self.env_id, hand_slot]
        else:
            pos = eef_pos[self.env_id]
            quat = eef_quat[self.env_id]
        return torch.cat([pos, quat], dim=0).to(self.env.device)

    # ==================================================================
    # Geometry helpers
    # ==================================================================

    def _shift_along_local_z(
        self, pose7: torch.Tensor, offset: float, backward: bool = True
    ) -> torch.Tensor:
        device = self.env.device
        pose7 = pose7.to(device)
        rot = quat_to_rot_matrix(pose7[3:7].unsqueeze(0))[0]
        approach = rot[:, 2]
        approach = approach / torch.norm(approach)
        delta = approach * offset
        new_pos = pose7[:3] - delta if backward else pose7[:3] + delta
        return torch.cat([new_pos, pose7[3:7]], dim=0)

    def _shift_world_z(self, pose7: torch.Tensor, dz: float) -> torch.Tensor:
        out = pose7.clone()
        out[2] += dz
        return out

    # ==================================================================
    # Phase 1: pick left grasp via single-arm IK (left = hand_id 1)
    # ==================================================================

    def _start_left_grasp_ik(self) -> None:
        env_obj_pose = self.env.get_object_world_pose(
            self.env_id, obj_name=self.obj_name, obj_id=int(self.obj_id)
        )
        # LEFT grasp pool filtered by part (default "body" — sturdy and
        # leaves the handle free for the right arm to grasp later).
        local_pool = self.env.get_grasp_pool(
            self.env_id,
            obj_name=self.obj_name,
            obj_id=int(self.obj_id),
            transform_to_world=False,
            part=self.left_part,
        )
        if local_pool is None or env_obj_pose is None or local_pool.shape[0] == 0:
            self.current_state = f"failed: no grasp pool for part={self.left_part!r}"
            return
        self._grasp_pool_local = local_pool.to(self.env.device)
        # Transform the local pool to world for the left IK.
        world_pool = self.env.transform_pool_by_object_pose(
            self._grasp_pool_local, env_obj_pose[:3], env_obj_pose[3:7]
        )
        self._grasp_pool_world = world_pool

        # Single-arm pack: hand_id=1 (left). pack_single_arm_goalset puts
        # NaN in slot 0 → right disabled for this solve.
        self.hand_id = 1
        target = self.pack_single_arm_goalset(world_pool)
        is_dual = bool(getattr(self.ik_server, "dual_mode", False))
        if is_dual:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=self._robot_state_dict(),
                mode="goalset",
                lock_base=False,
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=self._robot_state_dict(),
                mode="goalset",
            )
        self._ik_token += 1
        self._ik_job = {
            "kind": "left_grasp",
            "token": self._ik_token,
            "future": self.ik_server.submit_ik(req),
        }
        self.current_state = "computing"

    def _poll_left_grasp_ik(self) -> bool:
        if self.left_grasp_world is not None:
            return True
        if self._ik_job is None or self._ik_job.get("kind") != "left_grasp":
            return False
        fut: concurrent.futures.Future = self._ik_job["future"]
        if not fut.done():
            return False
        try:
            success_list, idx_list, _ = fut.result()
        except Exception as ex:
            log.error("[Handover] left grasp IK exception: {}", ex)
            self.current_state = f"failed: left ik exception {ex}"
            self._ik_job = None
            return False
        ok = bool(success_list[0]) if success_list else False
        idx = int(idx_list[0]) if idx_list else -1
        if not ok or idx < 0 or idx >= self._grasp_pool_world.shape[0]:
            log.error(
                "[Handover] left grasp IK no solution: success={} idx={}",
                success_list,
                idx_list,
            )
            self.current_state = "failed: left ik no solution"
            self._ik_job = None
            return False
        self.left_grasp_local = self._grasp_pool_local[idx].clone()
        self.left_grasp_world = self._grasp_pool_world[idx].clone()
        self.left_pre_grasp_world = self._shift_along_local_z(
            self.left_grasp_world, self.pre_grasp_offset, backward=True
        )
        self.left_lift_world = self._shift_world_z(
            self.left_grasp_world, self.lift_offset
        )
        log.info(
            "[Handover] env_id={} chose left grasp idx={}/{} world=({:+.3f},{:+.3f},{:+.3f})",
            self.env_id,
            idx,
            self._grasp_pool_world.shape[0],
            *self.left_grasp_world[:3].cpu().tolist(),
        )
        self._ik_job = None
        self.current_state = "ready"
        return True

    # ==================================================================
    # Phase 5: paired handover IK
    # ==================================================================

    def _start_handover_ik(self) -> None:
        device = self.env.device
        # RIGHT grasp pool filtered by part (default "handle" — the
        # functional grasp; ~22 candidates for mug 8684, all kept since
        # max_right_pool=22). If a different part is configured, the
        # standard subsample-by-stride keeps the goalset bounded.
        right_local = self.env.get_grasp_pool(
            self.env_id,
            obj_name=self.obj_name,
            obj_id=int(self.obj_id),
            transform_to_world=False,
            part=self.right_part,
        )
        if right_local is None or right_local.shape[0] == 0:
            self.current_state = (
                f"failed: no right grasp pool for part={self.right_part!r}"
            )
            return
        if right_local.shape[0] > self.max_right_pool:
            stride = max(1, right_local.shape[0] // self.max_right_pool)
            right_local = right_local[::stride][: self.max_right_pool]
        right_local = right_local.to(device)
        n_r = right_local.shape[0]

        # Mug-pose sweep mirrors open-loop: 432 candidates by default
        # (3 z × 3 y × 1 x × 8 yaw × 3 pitch × 3 roll).
        mug_poses = self.env.generate_handover_mug_poses(
            n_yaws=self.n_yaws,
            pitch_degs=self.pitch_degs,
            roll_degs=self.roll_degs,
        ).to(device)
        n_m = mug_poses.shape[0]
        if n_r == 0 or n_m == 0:
            self.current_state = "failed: empty handover pool"
            return

        # Build cross-product (handover_mug × right_local) in world frame.
        rights_world = torch.empty((n_m * n_r, 7), device=device, dtype=torch.float32)
        lefts_world = torch.empty((n_m * n_r, 7), device=device, dtype=torch.float32)
        meta: List[tuple] = []
        # Could vectorize, but the loop is once per skill and N ~ 1000.
        left_local = self.left_grasp_local.to(device)
        for mi in range(n_m):
            mug_pos = mug_poses[mi, :3]
            mug_quat = mug_poses[mi, 3:7]
            left_world = RigidObject.transform_pose_to_world(
                left_local, mug_pos, mug_quat
            )
            for ri in range(n_r):
                row = mi * n_r + ri
                rights_world[row] = RigidObject.transform_pose_to_world(
                    right_local[ri], mug_pos, mug_quat
                )
                lefts_world[row] = left_world
                meta.append((mi, ri))

        # Paired goalset (1, G, L=2, 7): slot 0 right, slot 1 left.
        G = rights_world.shape[0]
        target = torch.empty((1, G, 14), device=device, dtype=torch.float32)
        target[0, :, :7] = rights_world
        target[0, :, 7:] = lefts_world

        is_dual = bool(getattr(self.ik_server, "dual_mode", False))
        if is_dual:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=self._robot_state_dict(),
                mode="goalset",
                lock_base=False,
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=self._robot_state_dict(),
                mode="goalset",
            )
        self._ik_token += 1
        self._ik_job = {
            "kind": "handover",
            "token": self._ik_token,
            "rights": rights_world,
            "lefts": lefts_world,
            "meta": meta,
            "n_mug": n_m,
            "n_right": n_r,
            # Sub-sampled local pool kept around so post-IK we can
            # recover the chosen right_local pose by index.
            "right_local_pool": right_local,
            "future": self.ik_server.submit_ik(req),
        }
        self.current_state = "computing"
        log.info(
            "[Handover] env_id={} handover paired IK submitted: "
            "n_mug_poses={} n_right_local={} G={}",
            self.env_id,
            n_m,
            n_r,
            G,
        )

    def _poll_handover_ik(self) -> bool:
        if self.right_grasp_world is not None:
            return True
        if self._ik_job is None or self._ik_job.get("kind") != "handover":
            return False
        fut: concurrent.futures.Future = self._ik_job["future"]
        if not fut.done():
            return False
        try:
            success_list, idx_list, _ = fut.result()
        except Exception as ex:
            log.error("[Handover] handover IK exception: {}", ex)
            self.current_state = f"failed: handover ik exception {ex}"
            self._ik_job = None
            return False
        ok = bool(success_list[0]) if success_list else False
        idx = int(idx_list[0]) if idx_list else -1
        rights = self._ik_job["rights"]
        if not ok or idx < 0 or idx >= rights.shape[0]:
            log.error(
                "[Handover] handover paired IK failed: G={} success={} idx={}",
                rights.shape[0],
                success_list,
                idx_list,
            )
            self.current_state = "failed: handover paired ik no solution"
            self._ik_job = None
            return False
        lefts = self._ik_job["lefts"]
        meta = self._ik_job["meta"][idx]
        # Save the chosen right-local pose so subsequent phases can
        # re-derive the right grasp target from the LATEST mug world
        # pose (drift correction — see _refresh_right_targets).
        self.chosen_right_local = self._ik_job["right_local_pool"][meta[1]].clone()
        self.right_grasp_world = rights[idx].clone()
        self.left_handover_eef = lefts[idx].clone()
        self.right_pre_grasp_world = self._shift_along_local_z(
            self.right_grasp_world, self.right_pre_grasp_offset, backward=True
        )
        # Left retract: move left back along ITS local -z (away from mug).
        self.left_retract_world = self._shift_along_local_z(
            self.left_handover_eef, self.retract_offset, backward=True
        )
        log.info(
            "[Handover] env_id={} handover chosen idx={}/{} mug_pose_idx={} "
            "right_local_idx={} R=({:+.3f},{:+.3f},{:+.3f}) "
            "L=({:+.3f},{:+.3f},{:+.3f})",
            self.env_id,
            idx,
            rights.shape[0],
            meta[0],
            meta[1],
            *self.right_grasp_world[:3].cpu().tolist(),
            *self.left_handover_eef[:3].cpu().tolist(),
        )
        self._ik_job = None
        self.current_state = "ready"
        return True

    # ==================================================================
    # Drift correction: refresh right targets from current mug pose
    # ==================================================================

    def _refresh_right_targets(self) -> bool:
        """Re-derive right grasp + left hold from current mug world pose.

        Called at the start of each right-arm phase. Reason: the left
        wrist never reaches its planned target exactly; the mug is held
        rigidly so its actual pose ≈ ``left_eef_actual ⊙
        inverse(left_grasp_local)`` and drifts a few mm/cm from the
        plan. Without this refresh, right closes on a stale ghost pose.
        """
        if self.chosen_right_local is None or self.left_grasp_local is None:
            return False
        mug_now = self.env.get_object_world_pose(
            self.env_id,
            obj_name=self.obj_name,
            obj_id=int(self.obj_id) if self.obj_id is not None else 0,
        )
        if mug_now is None:
            return False
        device = self.env.device
        mug_pos = mug_now[:3].to(device)
        mug_quat = mug_now[3:7].to(device)
        self.right_grasp_world = RigidObject.transform_pose_to_world(
            self.chosen_right_local.to(device), mug_pos, mug_quat
        )
        self.left_handover_eef = RigidObject.transform_pose_to_world(
            self.left_grasp_local.to(device), mug_pos, mug_quat
        )
        self.right_pre_grasp_world = self._shift_along_local_z(
            self.right_grasp_world, self.right_pre_grasp_offset, backward=True
        )
        self.left_retract_world = self._shift_along_local_z(
            self.left_handover_eef, self.retract_offset, backward=True
        )
        return True

    # ==================================================================
    # Action builders
    # ==================================================================

    def _left_action_8d(self, pose7: torch.Tensor, grip: float) -> torch.Tensor:
        device = self.env.device
        return torch.cat(
            [
                pose7.to(device),
                torch.tensor([grip], device=device, dtype=torch.float32),
            ],
            dim=0,
        )

    def _dual_action_16d(
        self,
        right_pose: torch.Tensor,
        left_pose: torch.Tensor,
        right_grip: float,
        left_grip: float,
    ) -> torch.Tensor:
        device = self.env.device
        gripper = torch.tensor(
            [right_grip, left_grip], device=device, dtype=torch.float32
        )
        return torch.cat([right_pose.to(device), left_pose.to(device), gripper], dim=0)

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def reset(self, action: List[Any]):
        # action: ["Handover", robot_id, obj_type, obj_name, obj_id]
        self.robot_id = int(action[1])
        self.obj_type = action[2]
        self.obj_name = action[3]
        self.obj_id = action[4]
        self.current_command = list(action)
        self.current_state = "ready"
        self.current_phase = "left_pre_grasp"

        self.left_grasp_local = None
        self.left_grasp_world = None
        self.left_pre_grasp_world = None
        self.left_lift_world = None
        self.right_grasp_world = None
        self.right_pre_grasp_world = None
        self.left_handover_eef = None
        self.left_retract_world = None
        self.chosen_right_local = None
        self._refreshed_for_phase = None
        self._ik_job = None
        self._grasp_pool_local = None
        self._grasp_pool_world = None
        # Capture right hold-pose AT RESET (env has settled by now via
        # the Task's init_limit). Stays fixed through every left-only
        # phase so the right wrist never sees a moving target.
        self.right_hold = self._get_eef_pose_for_arm(0).clone()

        self.planner_manager = self._get_planner_manager()
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            raise RuntimeError("IKServer not available in PlannerManager.")
        name = self._resolve_robot_name()
        if name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{name}'.")
        self.ik_server = ik_dict[name]

        # Update obstacles, ignoring the target object so the grasp doesn't
        # collide with the mug itself.
        if self.obj_name is not None:
            self.planner_manager.update_obstacles(
                obstacle_avoidance_path_list=["dynamic"],
                env_ids=[self.env_id],
                obstacle_ignore_path_list=[self.obj_name],
            )

        self._start_left_grasp_ik()

    def refresh(self, action: List[Any]):
        new_cmd = list(action)
        if self.current_command != new_cmd or self.current_phase is None:
            self.reset(action)

    # ==================================================================
    # Step: emit action for the current phase
    # ==================================================================

    def step(self):
        if self.current_state and self.current_state.startswith("failed"):
            self.current_action = "Failed"
            return "Failed"

        # Phase 1 needs the IK result before any MoveL fires.
        if self.left_grasp_world is None:
            if not self._poll_left_grasp_ik():
                if self.current_state and self.current_state.startswith("failed"):
                    self.current_action = "Failed"
                    return "Failed"
                self.current_action = None
                return None

        # Phase 5 (compute_handover) is also a pure IK wait.
        if self.current_phase == "compute_handover" and self.right_grasp_world is None:
            if self._ik_job is None:
                self._start_handover_ik()
            if not self._poll_handover_ik():
                if self.current_state and self.current_state.startswith("failed"):
                    self.current_action = "Failed"
                    return "Failed"
                self.current_action = None
                return None

        self.current_state = "running"

        # Drift correction: at the FIRST step of each right-arm phase
        # (EXCLUDING ``right_pre_grasp``), re-derive right_grasp_world /
        # left_handover_eef from the mug's CURRENT world pose. The chosen
        # (right_local, left_local) saved at compute_handover defines the
        # relative offsets — those are invariants of the grasp; only the
        # mug's drifted world pose changes between plan time and exec time.
        #
        # ``right_pre_grasp`` is intentionally excluded: at that phase
        # the LEFT arm hasn't yet swung to the handover pose, so the mug
        # is still at its lift position. Refreshing here would tell the
        # left arm to "stay where you are" instead of moving to handover,
        # and the right arm would chase a stale mug pose. Use the planned
        # IK values for that one phase; refresh kicks in once the left
        # has actually reached the handover spot.
        right_phases_needing_refresh = {
            "right_grasp",
            "left_retract",
        }
        if (
            self.current_phase in right_phases_needing_refresh
            and self._refreshed_for_phase != self.current_phase
        ):
            if self._refresh_right_targets():
                self._refreshed_for_phase = self.current_phase

        if self.current_phase == "left_pre_grasp":
            target = self._dual_action_16d(
                self.right_hold,
                self.left_pre_grasp_world,
                right_grip=0.0,
                left_grip=0.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "left_grasp":
            target = self._dual_action_16d(
                self.right_hold,
                self.left_grasp_world,
                right_grip=0.0,
                left_grip=0.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "left_close":
            # MoveL with right latched at right_hold + left at grasp +
            # left grip flipped to 1. Mirrors open-loop's "every step
            # is a 16D MoveL with grip baked in" pattern.
            target = self._dual_action_16d(
                self.right_hold,
                self.left_grasp_world,
                right_grip=0.0,
                left_grip=1.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "left_lift":
            target = self._dual_action_16d(
                self.right_hold,
                self.left_lift_world,
                right_grip=0.0,
                left_grip=1.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "compute_handover":
            # Should have been handled above; if we got here right_grasp_world
            # must be set, but no MoveL is needed for this synthetic phase.
            # Advance immediately by emitting None — update() will tick.
            self.current_action = None
            return None

        if self.current_phase == "right_pre_grasp":
            target = self._dual_action_16d(
                self.right_pre_grasp_world,
                self.left_handover_eef,
                right_grip=0.0,
                left_grip=1.0,
            )
            # planner_mode=1: FORCE MotionGen (collision-aware planning)
            # for the longest swing of the whole skill — left arm goes
            # from lift pose to the handover spot, right arm reaches
            # in from across the table. ServoL would jolt both wrists
            # along straight lines, easily knocking the mug out of the
            # left gripper or driving the right elbow into self-collision.
            # MotionGen plans a smooth feasible trajectory respecting
            # joint limits and obstacles.
            self.current_action = {"MoveL": ((self.robot_id, -1, 1), target)}
            return self.current_action

        if self.current_phase == "right_grasp":
            target = self._dual_action_16d(
                self.right_grasp_world,
                self.left_handover_eef,
                right_grip=0.0,
                left_grip=1.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "right_close":
            # MoveL hand_id=-1 latches both arms at handover targets;
            # right_grip flips 0→1 to close. Same pattern as open-loop.
            target = self._dual_action_16d(
                self.right_grasp_world,
                self.left_handover_eef,
                right_grip=1.0,
                left_grip=1.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "left_open":
            # Both arms latched, left_grip flips 1→0. Right keeps the mug.
            target = self._dual_action_16d(
                self.right_grasp_world,
                self.left_handover_eef,
                right_grip=1.0,
                left_grip=0.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        if self.current_phase == "left_retract":
            target = self._dual_action_16d(
                self.right_grasp_world,
                self.left_retract_world,
                right_grip=1.0,
                left_grip=0.0,
            )
            self.current_action = {"MoveL": ((self.robot_id, -1, -1), target)}
            return self.current_action

        self.current_state = "failed: bad phase"
        self.current_action = "Failed"
        return "Failed"

    # ==================================================================
    # Update: phase progression
    # ==================================================================

    def update(self, info):
        base = {
            "atomic_skill_type": "Handover",
            "command": self.current_command,
            "action": self.current_action,
            "phase": self.current_phase,
        }

        if self.current_state == "computing":
            return {**base, "finished": False, "state": "computing", "truncated": 0}
        if self.current_state and self.current_state.startswith("failed"):
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        # compute_handover has no MoveL — once IK is solved, advance.
        if self.current_phase == "compute_handover":
            if self.right_grasp_world is not None:
                self._advance_phase()
                return {
                    **base,
                    "finished": False,
                    "state": f"running: {self.current_phase}",
                    "truncated": 0,
                    "phase": self.current_phase,
                }
            return {**base, "finished": False, "state": "computing", "truncated": 0}

        gp = info.get("global_planner_info", None)
        if gp is None or gp[self.env_id] is None:
            return {
                **base,
                "finished": False,
                "state": self.current_state or "ready",
                "truncated": 0,
            }
        env_gp = gp[self.env_id]
        trunc = env_gp.get("truncated", 0)
        if trunc == 1:
            self.current_state = "truncated: env terminated first"
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        if trunc == 2:
            self.current_state = "truncated: env truncated first"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        if trunc == 3:
            self.current_state = "failed: global planner failed to plan"
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }

        if env_gp.get("finished", False):
            if self.current_phase == _PHASES[-1]:
                self.current_state = "finished"
                return {
                    **base,
                    "finished": True,
                    "state": "finished",
                    "truncated": 0,
                    "phase": "completed",
                }
            self._advance_phase()
            log.info("[Handover] env_id={} phase={}", self.env_id, self.current_phase)
            return {
                **base,
                "finished": False,
                "state": f"running: {self.current_phase}",
                "truncated": 0,
                "phase": self.current_phase,
            }

        return {**base, "finished": False, "state": "running", "truncated": 0}

    def _advance_phase(self) -> None:
        try:
            i = _PHASES.index(self.current_phase)
        except ValueError:
            i = -1
        if 0 <= i < len(_PHASES) - 1:
            self.current_phase = _PHASES[i + 1]
