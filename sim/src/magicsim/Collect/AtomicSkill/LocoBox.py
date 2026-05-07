"""
LocoBox atomic skill: bimanual ground-box squeeze + lift for the
``g1_fixed_hand`` robot (rigid rubber_hand — no finger DoFs).

Selection step (NEW):
    Before any phase runs we sample a paired goalset of candidate
    pre_grasp poses, submit it through paired IK (DualIKServer with
    ``mode="goalset"``), and let the solver pick which (translation,
    tilt) combo is reachable. Once selected, the same (forward_ratio,
    down_ratio, tilt_deg) tuple is reused to construct grasp + lift
    targets so the whole sequence stays consistent.

    Sampling space (config-driven):
      * forward_ratio in ``forward_ratio_range`` (n_forward samples)
      * down_ratio    in ``down_ratio_range``    (n_down samples)
      * tilt_deg      in ``tilt_deg_range``      (n_tilt samples)
        — wrist rotation about world Y axis ("pitch"); 0° = identity,
        positive tilts the palm forward/down for slanted insertion.
    Total goalset size G = n_forward × n_down × n_tilt.

Phase wiring (after selection):

* ``pre_grasp``: ``RetractMoveL`` — free-base locomotion. Walks to the
  locked-XY anchor that ``g1_fixed_hand_move_strategy`` computes
  (``lock_fwd_offset = 0.45`` behind the grasp).
* ``bend``:      ``MobileMoveL`` mode 4 — locked base, EEF linear-interp,
  no MotionGen, no IK server. Lerps the wrist target down to the low
  pre_grasp pose so the runtime Pink IK (with widened waist URDF) does
  the actual torso bend. Locked-base MotionGen disabled here — its
  planning is unreliable for deep waist pitches and adds latency.
* ``squeeze``:   ``MobileMoveL`` mode 4 — locked base, EEF linear-interp
  inward to the squeeze pose (selected orientation, y_local = ±(hy − gap)).
* ``lift``:      ``MobileMoveL`` mode 4 — vertical lift via lerp.

Env contract: ``get_target_bbox_half_extents(env_id, obj_name, obj_id)``
and ``get_target_world_pose(env_id, obj_name, obj_id)``. Both implemented
in :class:`magicsim.Task.LocoManip.Env.LocoBoxEnv.LocoBoxEnv`.
"""

import math
from typing import Any, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Task.LocoManip.Env.Test.TestLocoGraspEnv import visualize_grasp_pose


def _quat_world_y(theta_rad: float) -> Tuple[float, float, float, float]:
    """wxyz quaternion for a rotation of ``theta_rad`` about world Y axis.

    Y axis was chosen because ``squeeze`` closes inward along ±y; rotating
    the wrist about Y tilts the palm forward/back without changing the
    closing axis. Positive θ tilts the local +z (palm "up") toward +x
    (forward).
    """
    half = 0.5 * theta_rad
    return (math.cos(half), 0.0, math.sin(half), 0.0)


class LocoBox(AtomicSkill):
    """Goalset-IK selected pre_grasp + RetractMoveL → bend → squeeze → lift."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.hand_id = -1  # bimanual

        # Geometry / phase knobs (ratios are fractions of the box half-extents).
        self.gap = float(getattr(config, "gap", 0.02))
        self.pre_gap = float(getattr(config, "pre_gap", 0.15))
        self.lift_height = float(getattr(config, "lift_height", 0.25))

        # Goalset sampling — translation ranges in box-local frame ratios,
        # tilt range in degrees. Defaults span what the rigid-hand robot
        # can plausibly reach on a ground-level box.
        fwd_range = list(getattr(config, "forward_ratio_range", [-0.8, -0.2]))
        down_range = list(getattr(config, "down_ratio_range", [0.0, 0.5]))
        tilt_range = list(getattr(config, "tilt_deg_range", [0.0, 75.0]))
        self.forward_ratio_range: Tuple[float, float] = (
            float(fwd_range[0]),
            float(fwd_range[1]),
        )
        self.down_ratio_range: Tuple[float, float] = (
            float(down_range[0]),
            float(down_range[1]),
        )
        self.tilt_deg_range: Tuple[float, float] = (
            float(tilt_range[0]),
            float(tilt_range[1]),
        )
        self.n_forward_samples = int(getattr(config, "n_forward_samples", 5))
        self.n_down_samples = int(getattr(config, "n_down_samples", 5))
        self.n_tilt_samples = int(getattr(config, "n_tilt_samples", 5))
        self.debug = bool(getattr(config, "debug", False))
        # Per-phase axis viz (right + left wrist target). Same convention as
        # BiDexGrasp's ``_viz_grasp_for_phase``.
        self.viz_grasp = bool(getattr(config, "viz_grasp", True))
        self._last_viz_phase: str | None = None

        # Target / state cache.
        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None
        self.current_phase: str | None = None
        self.r_pre = self.l_pre = None
        self.r_grasp = self.l_grasp = None
        self.r_lift = self.l_lift = None

        # Async goalset-IK selection state.
        self._selected_params: Optional[Tuple[float, float, float]] = None
        self._goalset_params: Optional[List[Tuple[float, float, float]]] = None
        self._goalset_future = None
        self._goalset_token = 0

        self.planner_manager = None
        self.ik_server = None
        self.robot_name: str | None = None

    # ------------------------------------------------------------------ helpers
    def _build_pair_local(
        self,
        hx: float,
        hy: float,
        hz: float,
        forward_ratio: float,
        down_ratio: float,
        tilt_deg: float,
        y_offset: float,  # signed offset added to ±hy: pre_gap (positive) or -gap (squeeze)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(right_local_7d, left_local_7d)`` in box frame."""
        device = self.env.device
        x_local = forward_ratio * hx
        z_local = -down_ratio * hz
        quat = _quat_world_y(math.radians(tilt_deg))
        y_local = hy + y_offset
        r_local = torch.tensor(
            [x_local, -y_local, z_local, *quat], device=device, dtype=torch.float32
        )
        l_local = torch.tensor(
            [x_local, +y_local, z_local, *quat], device=device, dtype=torch.float32
        )
        return r_local, l_local

    def _to_world_pair(
        self,
        r_local: torch.Tensor,
        l_local: torch.Tensor,
        box_pos: torch.Tensor,
        box_quat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            RigidObject.transform_pose_to_world(r_local, box_pos, box_quat),
            RigidObject.transform_pose_to_world(l_local, box_pos, box_quat),
        )

    def _get_box_geometry(
        self,
    ) -> Optional[Tuple[Tuple[float, float, float], torch.Tensor, torch.Tensor]]:
        """Return ``(half_extents, box_pos, box_quat)`` or None on failure."""
        if not hasattr(self.env, "get_target_bbox_half_extents") or not hasattr(
            self.env, "get_target_world_pose"
        ):
            self.current_state = "failed: env lacks bbox / pose helpers"
            return None
        half = self.env.get_target_bbox_half_extents(
            env_id=self.env_id, obj_name=self.obj_name, obj_id=int(self.obj_id)
        )
        if half is None:
            self.current_state = "failed: bbox unavailable"
            return None
        pose = self.env.get_target_world_pose(
            env_id=self.env_id, obj_name=self.obj_name, obj_id=int(self.obj_id)
        )
        if pose is None:
            self.current_state = "failed: target world pose unavailable"
            return None
        device = self.env.device
        box_pos = pose[:3].to(device)
        box_quat = pose[3:7].to(device)
        return (
            (float(half[0]), float(half[1]), float(half[2])),
            box_pos,
            box_quat,
        )

    # ------------------------------------------------------------------ goalset
    def _sample_paired_pre_grasp_goalset(self) -> Optional[torch.Tensor]:
        """Build ``(G, 14)`` paired goalset of candidate pre_grasp poses
        (world frame). Stores per-row params in ``self._goalset_params`` so
        we can recover the chosen (forward_ratio, down_ratio, tilt_deg)
        after IK selection.
        """
        geo = self._get_box_geometry()
        if geo is None:
            return None
        (hx, hy, hz), box_pos, box_quat = geo
        if self.n_forward_samples > 1:
            fr_grid = torch.linspace(*self.forward_ratio_range, self.n_forward_samples)
        else:
            fr_grid = torch.tensor([sum(self.forward_ratio_range) / 2])
        if self.n_down_samples > 1:
            dr_grid = torch.linspace(*self.down_ratio_range, self.n_down_samples)
        else:
            dr_grid = torch.tensor([sum(self.down_ratio_range) / 2])
        if self.n_tilt_samples > 1:
            tilt_grid = torch.linspace(*self.tilt_deg_range, self.n_tilt_samples)
        else:
            tilt_grid = torch.tensor([self.tilt_deg_range[0]])

        rows: List[torch.Tensor] = []
        params: List[Tuple[float, float, float]] = []
        for fr in fr_grid:
            for dr in dr_grid:
                for tilt in tilt_grid:
                    r_local, l_local = self._build_pair_local(
                        hx,
                        hy,
                        hz,
                        float(fr),
                        float(dr),
                        float(tilt),
                        y_offset=self.pre_gap,
                    )
                    r_world, l_world = self._to_world_pair(
                        r_local, l_local, box_pos, box_quat
                    )
                    rows.append(torch.cat([r_world, l_world], dim=0))
                    params.append((float(fr), float(dr), float(tilt)))

        if not rows:
            self.current_state = "failed: empty goalset"
            return None
        self._goalset_params = params
        return torch.stack(rows, dim=0)

    def _submit_paired_goalset(self, goalset: torch.Tensor):
        """``goalset`` shape ``(G, 14)``. Returns the IK future."""
        robot_state = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        if isinstance(robot_state, dict):
            rs = robot_state.get(self.robot_name, next(iter(robot_state.values())))
        else:
            rs = robot_state
        robot_states_dict = {
            "base_pos": rs["base_pos"],
            "base_quat": rs["base_quat"],
            "joint_pos": rs["joint_pos"],
            "joint_vel": rs["joint_vel"],
        }
        target = (
            goalset.to(device=self.env.device).unsqueeze(0).contiguous()
        )  # (1, G, 14)
        is_dual = bool(getattr(self.ik_server, "dual_mode", False))
        if is_dual:
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

    def _kickoff_goalset_selection(self) -> bool:
        goalset = self._sample_paired_pre_grasp_goalset()
        if goalset is None:
            return False
        self._goalset_token += 1
        try:
            self._goalset_future = self._submit_paired_goalset(goalset)
        except Exception as ex:
            self.current_state = f"failed: paired IK submit exception: {ex}"
            return False
        return True

    def _poll_goalset_result(self) -> str:
        """Returns ``"pending" | "ready" | "failed"``."""
        fut = self._goalset_future
        if fut is None:
            return "failed"
        if not fut.done():
            return "pending"
        try:
            success_list, idx_list, env_ids = fut.result()
        except Exception as ex:
            self.current_state = f"failed: paired IK exception: {ex}"
            self._goalset_future = None
            return "failed"
        self._goalset_future = None
        if not env_ids or int(env_ids[0]) != int(self.env_id):
            self.current_state = "failed: paired IK env mismatch"
            return "failed"
        if not (len(success_list) >= 1 and bool(success_list[0])):
            self.current_state = "failed: paired IK no solution"
            return "failed"
        idx = int(idx_list[0]) if idx_list and len(idx_list) >= 1 else -1
        if idx < 0 or self._goalset_params is None or idx >= len(self._goalset_params):
            self.current_state = "failed: paired IK invalid index"
            return "failed"
        self._selected_params = self._goalset_params[idx]
        return "ready"

    # ------------------------------------------------------------------ targets
    def _compute_targets_from_selected(self) -> bool:
        """Build pre_grasp / squeeze / lift world poses from the selected params."""
        if self._selected_params is None:
            self.current_state = "failed: no selected goalset params"
            return False
        geo = self._get_box_geometry()
        if geo is None:
            return False
        (hx, hy, hz), box_pos, box_quat = geo
        forward_ratio, down_ratio, tilt_deg = self._selected_params

        # pre_grasp: parked outside ±y faces by pre_gap
        r_local, l_local = self._build_pair_local(
            hx,
            hy,
            hz,
            forward_ratio,
            down_ratio,
            tilt_deg,
            y_offset=self.pre_gap,
        )
        self.r_pre, self.l_pre = self._to_world_pair(
            r_local, l_local, box_pos, box_quat
        )

        # squeeze: wrists past ±y faces by gap (negative offset)
        r_local, l_local = self._build_pair_local(
            hx,
            hy,
            hz,
            forward_ratio,
            down_ratio,
            tilt_deg,
            y_offset=-self.gap,
        )
        self.r_grasp, self.l_grasp = self._to_world_pair(
            r_local, l_local, box_pos, box_quat
        )

        # lift: squeeze + (0,0,lift_height) in world frame
        self.r_lift = self.r_grasp.clone()
        self.r_lift[2] += self.lift_height
        self.l_lift = self.l_grasp.clone()
        self.l_lift[2] += self.lift_height

        return True

    def _build_target_14d(
        self, right: torch.Tensor, left: torch.Tensor
    ) -> torch.Tensor:
        device = self.env.device
        return torch.cat([right.to(device), left.to(device)], dim=0)

    def _placeholder_target(self) -> torch.Tensor:
        return torch.full((14,), torch.nan, device=self.env.device, dtype=torch.float32)

    # ------------------------------------------------------------------ viz
    def _get_env_origin(self) -> torch.Tensor:
        origin = self.env.scene.env_origins[self.env_id].clone()
        if origin.ndim > 1:
            origin = origin[0]
        return origin.to(self.env.device)

    def _pose_to_world(self, pose7: torch.Tensor) -> torch.Tensor:
        """Targets are env-local; Isaac debug-draw lives in absolute world."""
        out = pose7.clone().to(self.env.device)
        out[:3] = out[:3] + self._get_env_origin()[:3]
        return out

    def _phase_pair(self, phase: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if phase == "pre_grasp" or phase == "bend":
            if self.r_pre is None or self.l_pre is None:
                return None
            return self.r_pre, self.l_pre
        if phase == "squeeze":
            if self.r_grasp is None or self.l_grasp is None:
                return None
            return self.r_grasp, self.l_grasp
        if phase == "lift":
            if self.r_lift is None or self.l_lift is None:
                return None
            return self.r_lift, self.l_lift
        return None

    def _viz_for_phase(self, phase: str | None) -> None:
        if not self.viz_grasp or phase is None:
            return
        pair = self._phase_pair(phase)
        if pair is None:
            return
        right, left = pair
        visualize_grasp_pose([self._pose_to_world(right), self._pose_to_world(left)])

    # ------------------------------------------------------------------ reset / refresh
    def _bind_planner(self) -> bool:
        self.planner_manager = getattr(self.env.scene, "planner_manager", None)
        if self.planner_manager is None:
            self.current_state = "failed: planner_manager not available"
            return False
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            self.current_state = "failed: ik_server not in planner_manager"
            return False
        names = list(ik_dict.keys())
        if self.robot_id < 0 or self.robot_id >= len(names):
            self.current_state = "failed: robot_id out of range"
            return False
        self.robot_name = names[self.robot_id]
        self.ik_server = ik_dict[self.robot_name]
        return True

    def reset(self, action: List[Any]):
        # action: [LocoBox, robot_id, obj_type, obj_name, obj_id]
        self.robot_id = int(action[1])
        self.hand_id = -1
        self.obj_type = action[2]
        self.obj_name = action[3]
        self.obj_id = action[4]

        self.current_state = "ready"
        self.current_command = list(action)
        # Always set a non-None phase. The upstream refresh path:
        #   ``if self.current_command != new_cmd or self.current_phase is None:``
        # would otherwise re-fire reset every tick during the init window
        # (obj_name=None), or while the async paired-IK is in flight —
        # discarding the future + restarting selection forever (and
        # spamming the debug log). ``"init"`` and ``"selecting"`` are
        # both inert sentinels; the real phase string starts at
        # ``"pre_grasp"`` once paired-IK selection is ready.
        self.current_phase = "init"
        self._last_viz_phase = None
        self._selected_params = None
        self._goalset_params = None
        self._goalset_future = None
        self.r_pre = self.l_pre = None
        self.r_grasp = self.l_grasp = None
        self.r_lift = self.l_lift = None

        if not self._bind_planner():
            return
        if self.obj_name is not None:
            self.planner_manager.update_obstacles(
                obstacle_avoidance_path_list=["dynamic"],
                env_ids=[self.env_id],
                obstacle_ignore_path_list=[self.obj_name],
            )
            if self._kickoff_goalset_selection():
                self.current_state = "selecting"
                self.current_phase = "selecting"

    def refresh(self, action: List[Any]):
        new_cmd = list(action)
        if self.current_command != new_cmd or self.current_phase is None:
            self.reset(action)

    # ------------------------------------------------------------------ step / update
    def step(self):
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        # Init phase: target obj not bound yet → emit placeholder.
        if self.obj_name is None or self.obj_id is None:
            target = self._placeholder_target()
            self.current_action = {
                "RetractMoveL": ((self.robot_id, -1, -1), target),
            }
            return self.current_action

        # Async paired-IK selection still in flight. Phase is "selecting"
        # purely as a sentinel for the refresh loop (see ``reset``);
        # ``current_state`` drives the actual gating.
        if self.current_state == "selecting":
            status = self._poll_goalset_result()
            if status == "pending":
                target = self._placeholder_target()
                self.current_action = {
                    "RetractMoveL": ((self.robot_id, -1, -1), target),
                }
                return self.current_action
            if status == "failed":
                self.current_action = "Failed"
                return "Failed"
            # status == "ready" → derive targets and start phase pipeline
            if not self._compute_targets_from_selected():
                self.current_action = "Failed"
                return "Failed"
            self.current_phase = "pre_grasp"
            self.current_state = "running"

        # Lazy fill if we somehow get here without computed targets.
        if self.r_pre is None or self.r_grasp is None or self.r_lift is None:
            if not self._compute_targets_from_selected():
                self.current_action = "Failed"
                return "Failed"

        # Refresh box pose every step (cheap — local pose lookup) so the
        # current phase's target follows the box if it shifts. Selected
        # (forward, down, tilt) are reused; only the box→world transform
        # changes, so this is always consistent with the IK selection.
        self._compute_targets_from_selected()

        # Per-phase viz: draw axes once when entering a new phase.
        if self.viz_grasp and self.current_phase != self._last_viz_phase:
            self._viz_for_phase(self.current_phase)
            self._last_viz_phase = self.current_phase

        # Phase dispatch:
        #   pre_grasp -> RetractMoveL (free base, planner_mode -1; picks up
        #               g1_fixed_hand move_strategy: lock_fwd_offset=0.45,
        #               clip_height=0.4).
        #   bend      -> MobileMoveL  (mode 4 = locked base + EEF lerp,
        #               no MG, no IK server. Same wrist target as pre_grasp;
        #               runtime Pink IK with widened waist does the bend
        #               while wrist lerps from current to target.)
        #   squeeze   -> MobileMoveL  (mode 4 = locked base, EEF lerp; box in
        #               obstacle ignore set, single-step snap would overshoot).
        #   lift      -> MobileMoveL  (mode 4 = vertical raise via lerp).
        if self.current_phase == "pre_grasp":
            target = self._build_target_14d(self.r_pre, self.l_pre)
            self.current_action = {
                "RetractMoveL": ((self.robot_id, -1, -1), target),
            }
        elif self.current_phase == "bend":
            target = self._build_target_14d(self.r_pre, self.l_pre)
            self.current_action = {
                "MobileMoveL": ((self.robot_id, -1, 4), target),
            }
        elif self.current_phase == "squeeze":
            target = self._build_target_14d(self.r_grasp, self.l_grasp)
            self.current_action = {
                "MobileMoveL": ((self.robot_id, -1, 4), target),
            }
        elif self.current_phase == "lift":
            target = self._build_target_14d(self.r_lift, self.l_lift)
            self.current_action = {
                "MobileMoveL": ((self.robot_id, -1, 4), target),
            }
        else:
            self.current_state = "failed"
            self.current_action = "Failed"
            return "Failed"

        self.current_state = "running"
        return self.current_action

    def update(self, info: dict) -> dict:
        base = {
            "atomic_skill_type": "LocoBox",
            "command": self.current_command,
            "action": self.current_action,
            "phase": self.current_phase,
        }
        if self.current_state == "failed" or (
            isinstance(self.current_state, str)
            and self.current_state.startswith("failed")
        ):
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        # While selecting we haven't entered a phase yet; report running.
        if self.current_state == "selecting":
            return {**base, "finished": False, "state": "selecting", "truncated": 0}

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
                # Re-snapshot box pose before bending (it may have settled
                # further during the walk).
                self._compute_targets_from_selected()
                self.current_phase = "bend"
                return {
                    **base,
                    "finished": False,
                    "state": "running: bend",
                    "truncated": 0,
                    "phase": "bend",
                }
            if self.current_phase == "bend":
                self.current_phase = "squeeze"
                return {
                    **base,
                    "finished": False,
                    "state": "running: squeeze",
                    "truncated": 0,
                    "phase": "squeeze",
                }
            if self.current_phase == "squeeze":
                self.current_phase = "lift"
                return {
                    **base,
                    "finished": False,
                    "state": "running: lift",
                    "truncated": 0,
                    "phase": "lift",
                }
            if self.current_phase == "lift":
                self.current_state = "finished"
                return {
                    **base,
                    "finished": True,
                    "state": "finished",
                    "truncated": 0,
                    "phase": "completed",
                }

        return {**base, "finished": False, "state": "running", "truncated": 0}
