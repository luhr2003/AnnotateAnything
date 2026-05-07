"""SonicPolicy — WBC policy wrapper for the SONIC hybrid IK pipeline.

This is structurally identical to the per-tick main loop of sonic's
`stage_hybrid_eval.py:332-423`, with Pink IK's upper-body solve replaced by
a read from `SonicArmBuffer` (filled externally by
`SonicPinkInverseKinematicsAction`).

    set_goal(action_5d) → unpack [vx, vy, ang_vel, height, mode_raw]
    set_observation({joint_pos, joint_vel, base_ang_vel, gravity_in_base,
                     root_quat_wxyz, arm_targets_17})

    get_action():
      - Derive planner cmd (see `_derive_planner_cmd` for the full contract)
      - Every 5 ticks: rebuild planner context from cache (or init) and run
        the batched planner ONNX; resample 30→50 Hz into `_planner_cache`
      - Sample 10 future frames from the cache: leg_pos/vel (MJ→IL scatter)
        and ref_root_quat
      - `joint_pos_future[..., upper_idx_il] ← arm_targets_17` (broadcast 10)
      - `SonicG1Inference.step(...)` → `target_il [N, 29]`
      - Advance `playback_idx` and `step_counter`; return `target_il`

==============================================================================
Planner-command semantics (see `_derive_planner_cmd` docstring for detail):

  * **vx, vy — world-frame** linear velocity commands (m/s). Planner was
    trained on world-frame unit-vector inputs, and the sonic pico teleop
    (`gear_sonic_deploy/.../gamepad.hpp:686`) feeds it world-frame unit
    vectors — we follow the same contract. `movement_dir` is just
    `[vx, vy, 0]` normalized; no body→world rotation.

  * **ang_vel — yaw rate** (rad/s) that feeds a **virtual accumulated world
    yaw** `self._facing_angle`, mirroring sonic pico's `planner_facing_angle`
    (gamepad.hpp:170, :388). `_facing_angle` starts at 0 (world +X), updates
    as `_facing_angle += ang_vel * dt` every tick, and is **decoupled from
    the robot's sim yaw** — the planner gets a stable externally-commanded
    heading and auto-corrects robot yaw back toward it. Reset (see
    `SonicPolicy.reset`) zeroes `_facing_angle`, matching pico gamepad.hpp:621.

  * **facing_dir** = `[cos(_facing_angle), sin(_facing_angle), 0]` (world unit)

  * **height, mode** — passed through to the planner directly; `mode=-1`
    (AUTO) auto-selects IDLE / SLOW_WALK / WALK / RUN by |v|.

==============================================================================
Upper-body (Pink IK) handoff:

Pink IK and SONIC are **decoupled through `SonicArmBuffer`**, not through
sim joint targets. Specifically:

  * `SonicPinkInverseKinematicsAction` every tick: (1) solves the 17-DOF
    waist+arm IK given an external rest-pose target, (2) writes the solution
    into `SonicArmBuffer` in `PINK_CONTROLLED_JOINTS_IL` order. It does
    **not** call `set_joint_position_target` — sim is not its concern.

  * `SonicPolicy.get_action` reads `arm_targets_17` from the buffer (set by
    the caller from `SonicArmBuffer.read_latest()`) and scatters it into
    `joint_pos_future[..., upper_idx_il]`, giving the SONIC encoder its
    upper-body reference.

  * `SonicWBCAction` writes **all 29 body DOF** from the decoder output to
    sim (`target_il` — legs + waist + arms, coordinated). The decoder was
    trained on coupled 29-DOF motion, so applying the whole thing preserves
    the learned lower/upper body coordination (arm swing during walking).

This mirrors `stage_hybrid_eval_magicsim.py:415-419`: Pink IK's solve only
informs the encoder; the decoder's 29-DOF is what physically drives the sim.

==============================================================================
Lazy init: `bind_articulation(articulation_data)` must be called once by
`SonicWBCAction.__init__` before the first `get_action`. It resolves body
joint indices, `action_scale = 0.25 * effort / stiffness`, and `default_angles`
from runtime articulation metadata, then builds the ORT sessions and the
`SonicG1Inference` ring buffers.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor  # noqa: F401  (kept for debug)
from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.assets.articulation.articulation_data import ArticulationData

from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.wbc_base_policy import (
    WBCPolicy,
)

from .G1.configs import (
    G1SonicV1Config,
    G1_BODY_JOINTS_IL,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_TO_ISAACLAB_DOF,
    LEG_JOINTS_IL,
    PINK_CONTROLLED_JOINTS_IL,
)
from .inference.planner_pool import (
    PLANNER_FRAME_DIM,
    PlannerSessionPool,
)
from .inference.quat_utils import (
    resample_traj_30_to_50hz,
    slerp_torch,
    RESAMPLED_FRAMES,
)
from .inference.sonic_g1_inference import SonicG1Inference
from .policy_constants import (
    ACTION_DIM,
    ALLOWED_PRED_NUM_TOKENS,
    AUTO_MODE_IDLE_MAX,
    AUTO_MODE_SLOW_WALK_MAX,
    AUTO_MODE_WALK_MAX,
    DT_POLICY,
    EPS_TARGET_VEL,
    G1_FUTURE_FRAME_SKIP,
    G1_NUM_FUTURE_FRAMES,
    HAND_JOINT_REGEX,
    LEG_JOINT_REGEXES,
    MODE_AUTO,
    MODE_IDLE,
    MODE_RUN,
    MODE_SLOW_WALK,
    MODE_WALK,
    NUM_BODY_DOF,
    OFFSET_ANG_VEL,
    OFFSET_HEIGHT,
    OFFSET_MODE,
    OFFSET_VX,
    OFFSET_VY,
    PLANNER_CONTEXT_DEFAULT_HEIGHT,
    PLANNER_EVERY_K_POLICY_STEPS,
)


# Sonic's training-time action_scale formula: 0.25 * effort_limit_sim / stiffness
# per joint. Mirrors the `G1_MODEL_12_ACTION_SCALE` constant dict in
# `gear_sonic/envs/manager_env/robots/g1.py:360-371`. Keys are regex patterns,
# values are floats. SonicPolicy uses these regexes to resolve the scale for
# each runtime body joint name.
_SONIC_G1_ACTION_SCALE_REGEX: dict[str, float] = {
    # Derived from actuator cfg params: 0.25 * E / K
    # legs
    r".*_hip_pitch_joint": 0.25 * 139.0 / (0.025101925 * (10 * 2 * np.pi) ** 2),
    r".*_hip_roll_joint": 0.25 * 139.0 / (0.025101925 * (10 * 2 * np.pi) ** 2),
    r".*_hip_yaw_joint": 0.25 * 88.0 / (0.01017752004 * (10 * 2 * np.pi) ** 2),
    r".*_knee_joint": 0.25 * 139.0 / (0.025101925 * (10 * 2 * np.pi) ** 2),
    # feet
    r".*_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * 0.003609725 * (10 * 2 * np.pi) ** 2),
    r".*_ankle_roll_joint": 0.25 * 50.0 / (2.0 * 0.003609725 * (10 * 2 * np.pi) ** 2),
    # waist
    r"waist_yaw_joint": 0.25 * 88.0 / (0.01017752004 * (10 * 2 * np.pi) ** 2),
    r"waist_roll_joint": 0.25 * 50.0 / (2.0 * 0.003609725 * (10 * 2 * np.pi) ** 2),
    r"waist_pitch_joint": 0.25 * 50.0 / (2.0 * 0.003609725 * (10 * 2 * np.pi) ** 2),
    # arms
    r".*_shoulder_pitch_joint": 0.25 * 25.0 / (0.003609725 * (10 * 2 * np.pi) ** 2),
    r".*_shoulder_roll_joint": 0.25 * 25.0 / (0.003609725 * (10 * 2 * np.pi) ** 2),
    r".*_shoulder_yaw_joint": 0.25 * 25.0 / (0.003609725 * (10 * 2 * np.pi) ** 2),
    r".*_elbow_joint": 0.25 * 25.0 / (0.003609725 * (10 * 2 * np.pi) ** 2),
    r".*_wrist_roll_joint": 0.25 * 25.0 / (0.003609725 * (10 * 2 * np.pi) ** 2),
    r".*_wrist_pitch_joint": 0.25 * 5.0 / (0.00425 * (10 * 2 * np.pi) ** 2),
    r".*_wrist_yaw_joint": 0.25 * 5.0 / (0.00425 * (10 * 2 * np.pi) ** 2),
}


def _action_scale_il(joint_names: list[str]) -> np.ndarray:
    """Regex-match each of the 29 body joint names against
    `_SONIC_G1_ACTION_SCALE_REGEX` and return a [29] float32 array. Raises
    if any joint fails to match."""
    scale = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        matched = False
        for pattern, val in _SONIC_G1_ACTION_SCALE_REGEX.items():
            if re.fullmatch(pattern, name):
                scale[i] = val
                matched = True
                break
        if not matched:
            raise ValueError(f"No sonic action_scale regex matched joint '{name}'")
    return scale


@dataclass
class _SonicCommand:
    """5D command cached by `set_goal`, unbound into per-component tensors."""

    vx: torch.Tensor  # [N]
    vy: torch.Tensor  # [N]
    ang_vel: torch.Tensor  # [N]
    height: torch.Tensor  # [N]
    mode_raw: torch.Tensor  # [N] long


class SonicPolicy(WBCPolicy):
    """SONIC hybrid IK pipeline WBC policy (produces a 29-DOF body target; the
    upper-body reference comes from `SonicArmBuffer`)."""

    wbc_config: G1SonicV1Config

    def __init__(
        self,
        wbc_config: G1SonicV1Config,
        num_envs: int = 1,
    ):
        self.wbc_config = wbc_config
        self.num_envs = int(num_envs)

        # Runtime state — all None until `bind_articulation` is called
        self._bound: bool = False
        self.device: torch.device | None = None

        self._body_idx_full: torch.Tensor | None = None  # [29] USD-full index
        self._body_joint_names: list[str] | None = None  # [29] names
        self._leg_idx_il: torch.Tensor | None = None  # [12]
        self._upper_idx_il: torch.Tensor | None = None  # [17]
        self._leg_mj_slots: torch.Tensor | None = (
            None  # [12] MJ slot holding each leg's value in planner output
        )
        self._il_to_mj: torch.Tensor | None = None  # [29]
        self._mj_to_il: torch.Tensor | None = None  # [29]
        self._upper_default_il: torch.Tensor | None = None  # [17] init_state arm pose
        self._default_jp_il: torch.Tensor | None = None  # [29] init state

        self._inference: SonicG1Inference | None = None
        self._planner_pool: PlannerSessionPool | None = None

        self._planner_context: torch.Tensor | None = None  # [N, 4, 36] MJ
        self._planner_cache: torch.Tensor | None = None  # [N, RESAMPLED_FRAMES, 36] MJ
        self._playback_idx: torch.Tensor | None = None  # [N]
        self._future_offsets: torch.Tensor | None = None  # [10]

        self._step_counter: int = 0

        # cache filled by set_goal / set_observation
        self._cmd: _SonicCommand | None = None
        self._obs: dict[str, torch.Tensor] | None = None

    # ==================================================================
    # Lazy bind to runtime articulation
    # ==================================================================
    def bind_articulation(self, articulation_data: ArticulationData) -> None:
        """Called once by `SonicWBCAction.__init__` after the articulation has
        spawned. Resolves body joint indices (dex fingers filtered out), the
        per-joint `action_scale`, and `default_jp_il`; then builds the ORT
        sessions (SonicG1Inference + PlannerSessionPool) and the planner cache.
        """
        assert not self._bound, "SonicPolicy.bind_articulation called twice"
        device = articulation_data.joint_pos.device
        self.device = device

        # 1. Resolve `body_idx_full` (filter out dex fingers from USD-full 43)
        full_joint_names = list(articulation_data.joint_names)
        hand_re = re.compile(HAND_JOINT_REGEX)
        body_idx_full: list[int] = [
            i for i, n in enumerate(full_joint_names) if not hand_re.fullmatch(n)
        ]
        body_joint_names = [full_joint_names[i] for i in body_idx_full]
        assert len(body_joint_names) == NUM_BODY_DOF, (
            f"Expected {NUM_BODY_DOF} body joints after filtering hand joints from "
            f"{len(full_joint_names)} USD joints, got {len(body_joint_names)}. "
            f"USD joints: {full_joint_names}"
        )
        # Invariant 2: filtered IL order must match sonic training IL order
        assert body_joint_names == G1_BODY_JOINTS_IL, (
            "Runtime body joint order differs from sonic training IL order.\n"
            f"Runtime: {body_joint_names}\nExpected: {G1_BODY_JOINTS_IL}"
        )
        self._body_idx_full = torch.as_tensor(
            body_idx_full, dtype=torch.long, device=device
        )
        self._body_joint_names = body_joint_names

        # 2. Resolve leg / upper IL indices + MJ permutations
        leg_patterns = [re.compile(p) for p in LEG_JOINT_REGEXES]
        leg_idx_il = [
            i
            for i, n in enumerate(body_joint_names)
            if any(p.fullmatch(n) for p in leg_patterns)
        ]
        upper_idx_il = [body_joint_names.index(n) for n in PINK_CONTROLLED_JOINTS_IL]
        assert len(leg_idx_il) == 12, f"expected 12 leg joints, got {len(leg_idx_il)}"
        assert set(leg_idx_il) | set(upper_idx_il) == set(range(NUM_BODY_DOF)), (
            "leg + pink-controlled indices do not cover all 29 body joints"
        )
        # Leg order sanity: must match LEG_JOINTS_IL exactly, else leg_mj_slots misaligns
        leg_names_resolved = [body_joint_names[i] for i in leg_idx_il]
        assert leg_names_resolved == LEG_JOINTS_IL, (
            f"Leg joints resolved in wrong order. Got {leg_names_resolved}, "
            f"expected {LEG_JOINTS_IL}"
        )

        self._leg_idx_il = torch.as_tensor(leg_idx_il, dtype=torch.long, device=device)
        self._upper_idx_il = torch.as_tensor(
            upper_idx_il, dtype=torch.long, device=device
        )
        self._il_to_mj = torch.as_tensor(
            G1_ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=device
        )
        self._mj_to_il = torch.as_tensor(
            G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=device
        )
        self._leg_mj_slots = self._mj_to_il[self._leg_idx_il]  # [12]

        # 3. default_angles (IL body 29) + action_scale (IL body 29)
        default_jp_full = (
            articulation_data.default_joint_pos[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        default_jp_il = default_jp_full[body_idx_full]  # [29]
        action_scale_il = _action_scale_il(body_joint_names)

        self._default_jp_il = torch.as_tensor(
            default_jp_il, dtype=torch.float32, device=device
        )
        self._upper_default_il = self._default_jp_il[self._upper_idx_il].clone()  # [17]

        # 4. Build SonicG1Inference + PlannerSessionPool
        ort_device_str = "cuda" if device.type == "cuda" else "cpu"
        self._inference = SonicG1Inference(
            num_envs=self.num_envs,
            g1_encoder_onnx=self.wbc_config.g1_encoder_onnx_path,
            decoder_onnx=self.wbc_config.decoder_onnx_path,
            default_angles=default_jp_il,
            action_scale=action_scale_il,
            device=ort_device_str,
            device_id=device.index if device.index is not None else 0,
        )
        self._planner_pool = PlannerSessionPool(
            model_path=self.wbc_config.planner_onnx_path,
            pool_size=self.num_envs,
            device_id=device.index if device.index is not None else 0,
            serial=False,
        )

        # 5. Planner state
        self._planner_context = torch.zeros(
            self.num_envs, 4, PLANNER_FRAME_DIM, device=device
        )
        self._planner_cache = torch.zeros(
            self.num_envs, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=device
        )
        self._playback_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._future_offsets = torch.arange(
            0,
            G1_NUM_FUTURE_FRAMES * G1_FUTURE_FRAME_SKIP,
            G1_FUTURE_FRAME_SKIP,
            dtype=torch.long,
            device=device,
        )

        # Virtual accumulated world yaw driving `facing_direction` — mirrors
        # sonic pico `planner_facing_angle` (gamepad.hpp:170). Starts at 0
        # (world +X) and integrates `ang_vel * dt` per tick. Decoupled from
        # the robot's actual sim yaw so the planner gets a stable, externally
        # commanded heading that it will track / correct toward.
        self._facing_angle = torch.zeros(self.num_envs, device=device)

        # Seed SonicG1Inference ring buffers with the default pose
        self._inference.reset(
            joint_pos=self._default_jp_il.unsqueeze(0).expand(self.num_envs, -1)
        )
        self._seed_planner_context_init()

        self._step_counter = 0
        self._bound = True

    # ==================================================================
    # Planner context helpers
    # ==================================================================
    def _seed_planner_context_init(self) -> None:
        """Seed the 4-frame planner context with the init_state stand pose
        (yaw-normalized, MJ joint order).

        Mirrors `stage_hybrid_eval.py:272-276`。
        """
        device = self.device
        default_jp_mj = self._default_jp_il[self._il_to_mj]  # [29]
        init_frame = torch.zeros(self.num_envs, PLANNER_FRAME_DIM, device=device)
        init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
        init_frame[:, 3] = 1.0  # quat w
        init_frame[:, 7:] = default_jp_mj.unsqueeze(0).expand(self.num_envs, -1)
        self._planner_context[:] = init_frame.unsqueeze(1).expand(-1, 4, -1).clone()
        self._planner_cache.zero_()
        self._playback_idx.zero_()

    def _context_from_cache(self) -> torch.Tensor:
        """Rebuild planner context from 4 historical cache frames (50 Hz → 30 Hz sample).

        Mirrors the deploy-side `UpdateContextFromMotion`
        (localmotion_kplanner.hpp:628-678) and `_context_from_cache` in
        `stage_hybrid_eval.py`.
        """
        cache = self._planner_cache
        L = cache.shape[1]
        device = cache.device
        t_offsets = torch.arange(4, device=device, dtype=torch.float32) * (50.0 / 30.0)
        idx_f = self._playback_idx.view(-1, 1).float() + t_offsets.view(1, -1)
        idx0 = idx_f.long().clamp(0, L - 1)
        idx1 = (idx0 + 1).clamp(0, L - 1)
        alpha = (idx_f - idx0.float()).clamp(0.0, 1.0)
        n_range = torch.arange(self.num_envs, device=device).view(-1, 1).expand(-1, 4)
        f0 = cache[n_range, idx0]
        f1 = cache[n_range, idx1]
        pos = f0[..., 0:3] + (f1[..., 0:3] - f0[..., 0:3]) * alpha.unsqueeze(-1)
        quat = slerp_torch(f0[..., 3:7], f1[..., 3:7], alpha)
        joints = f0[..., 7:] + (f1[..., 7:] - f0[..., 7:]) * alpha.unsqueeze(-1)
        return torch.cat([pos, quat, joints], dim=-1)

    def _run_planner(
        self,
        mode: torch.Tensor,  # [N] long
        movement_direction: torch.Tensor,  # [N, 3]
        facing_direction: torch.Tensor,  # [N, 3]
        target_vel: torch.Tensor,  # [N]
        height: torch.Tensor,  # [N]
    ) -> None:
        ctx = self._planner_context.cpu().numpy().astype(np.float32)
        mv = movement_direction.cpu().numpy().astype(np.float32)
        fd = facing_direction.cpu().numpy().astype(np.float32)
        md = mode.cpu().numpy().astype(np.int64)
        tv = target_vel.cpu().numpy().astype(np.float32)
        ht = height.cpu().numpy().astype(np.float32)
        allowed_mask = np.asarray(ALLOWED_PRED_NUM_TOKENS, dtype=np.int64).reshape(
            1, 11
        )

        feeds: list[dict[str, np.ndarray]] = []
        for i in range(self.num_envs):
            feeds.append(
                {
                    "context_mujoco_qpos": ctx[i : i + 1],  # [1, 4, 36]
                    "target_vel": tv[i : i + 1].reshape(1),
                    "mode": md[i : i + 1].reshape(1),
                    "movement_direction": mv[i : i + 1],  # [1, 3]
                    "facing_direction": fd[i : i + 1],  # [1, 3]
                    "random_seed": np.array([0], dtype=np.int64),
                    "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                    "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                    "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                    "allowed_pred_num_tokens": allowed_mask,
                    "height": ht[i : i + 1].reshape(1),
                }
            )
        traj, _ = self._planner_pool.run_batched(feeds)  # [N, 64, 36]
        traj_t = torch.as_tensor(traj, device=self.device, dtype=torch.float32)
        self._planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
        self._playback_idx.zero_()

    # ==================================================================
    # 5D action → sonic planner args
    # ==================================================================
    def _derive_planner_cmd(
        self,
        cmd: _SonicCommand,
        root_quat_wxyz: torch.Tensor,  # [N, 4]  (unused, kept for signature compat)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map the 5D user action to sonic planner inputs.

        **Semantic contract** (world-frame, aligned with sonic pico teleop
        `gamepad.hpp:385-405`):
          - `vx`, `vy` are **world-frame** linear velocity commands (m/s).
            `movement_dir` = `[vx, vy, 0]` normalized (unit vector); no rotation.
          - `facing_dir` is driven by a **virtual accumulated world yaw**
            `self._facing_angle`, a policy-side state variable that integrates
            `ang_vel * dt` each tick. This is **decoupled from the robot's sim
            yaw** — mirrors pico's `planner_facing_angle`. At reset it goes back
            to 0 (world +X).
          - `ang_vel` (rad/s) is the increment added to `_facing_angle` per
            tick, equivalent to pico's `planner_facing_angle -= 0.02 * rx` with
            `rx` replaced by `-ang_vel * POLICY_HZ` (sign swapped so positive
            `ang_vel` = yaw-increasing = CCW).

        The planner gets a world-frame command that does not drift with robot
        motion; it will auto-correct robot yaw/position toward the commanded
        trajectory.

        `root_quat_wxyz` is retained in the signature for future flexibility
        but is **not used** here.
        """
        del root_quat_wxyz  # intentionally unused — see docstring

        # Accumulate virtual world yaw driving `facing_direction`.
        self._facing_angle = self._facing_angle + cmd.ang_vel * DT_POLICY

        # facing_dir: world-frame unit vector from the accumulated facing angle.
        facing_dir = torch.stack(
            [
                torch.cos(self._facing_angle),
                torch.sin(self._facing_angle),
                torch.zeros_like(self._facing_angle),
            ],
            dim=-1,
        )  # [N, 3]

        # vx, vy are already **world-frame** components — normalize to a unit
        # vector for `movement_dir` (planner expects a unit vector; magnitude
        # goes into `target_vel`).
        target_vel = torch.hypot(cmd.vx, cmd.vy)  # [N]
        safe_tv = target_vel.clamp(min=EPS_TARGET_VEL)
        movement_dir_xy = torch.stack(
            [cmd.vx / safe_tv, cmd.vy / safe_tv, torch.zeros_like(cmd.vx)],
            dim=-1,
        )  # [N, 3]
        low_mask = (target_vel < EPS_TARGET_VEL).unsqueeze(-1)
        movement_dir = torch.where(low_mask, facing_dir, movement_dir_xy)

        # Auto mode: |v| thresholds
        auto_mode = torch.where(
            target_vel < AUTO_MODE_IDLE_MAX,
            torch.full_like(cmd.mode_raw, MODE_IDLE),
            torch.where(
                target_vel < AUTO_MODE_SLOW_WALK_MAX,
                torch.full_like(cmd.mode_raw, MODE_SLOW_WALK),
                torch.where(
                    target_vel < AUTO_MODE_WALK_MAX,
                    torch.full_like(cmd.mode_raw, MODE_WALK),
                    torch.full_like(cmd.mode_raw, MODE_RUN),
                ),
            ),
        )
        mode = torch.where(cmd.mode_raw == MODE_AUTO, auto_mode, cmd.mode_raw)

        return mode, movement_dir, facing_dir, target_vel, cmd.height

    # ==================================================================
    # WBCPolicy API
    # ==================================================================
    def set_goal(self, goal: dict) -> None:
        """`goal = {"action": torch.Tensor [N, 5]}`. See `policy_constants.py` for layout."""
        action = goal["action"]
        assert action.shape == (self.num_envs, ACTION_DIM), (
            f"SonicPolicy expects action of shape ({self.num_envs}, {ACTION_DIM}), "
            f"got {tuple(action.shape)}"
        )
        action = action.to(self.device)
        self._cmd = _SonicCommand(
            vx=action[:, OFFSET_VX],
            vy=action[:, OFFSET_VY],
            ang_vel=action[:, OFFSET_ANG_VEL],
            height=action[:, OFFSET_HEIGHT],
            mode_raw=action[:, OFFSET_MODE].long(),
        )

    def set_observation(self, observation: dict) -> None:
        """Expected keys: joint_pos, joint_vel, base_ang_vel, gravity_in_base,
        root_quat_wxyz, arm_targets_17 — assembled by `SonicWBCAction.process_actions`."""
        self._obs = observation

    def get_action(self, time: float | None = None) -> torch.Tensor:
        """Run one 50 Hz policy tick. Returns motor targets [N, 29] in **IL body** order."""
        assert self._bound, (
            "SonicPolicy.bind_articulation must be called before get_action"
        )
        assert self._cmd is not None, (
            "SonicPolicy.set_goal not called before get_action"
        )
        assert self._obs is not None, (
            "SonicPolicy.set_observation not called before get_action"
        )
        del time

        joint_pos_il = self._obs["joint_pos"]  # [N, 29]
        joint_vel_il = self._obs["joint_vel"]  # [N, 29]
        base_ang_vel = self._obs["base_ang_vel"]  # [N, 3]
        gravity_in_base = self._obs["gravity_in_base"]  # [N, 3]
        root_quat_wxyz = self._obs["root_quat_wxyz"]  # [N, 4]
        arm_targets_17 = self._obs[
            "arm_targets_17"
        ]  # [N, 17] in PINK_CONTROLLED_JOINTS_IL order

        # 1. 5D action → planner args
        mode, move_dir, face_dir, target_vel, height = self._derive_planner_cmd(
            self._cmd, root_quat_wxyz
        )

        # 2. Run planner every 5 policy ticks
        if self._step_counter % PLANNER_EVERY_K_POLICY_STEPS == 0:
            if self._step_counter > 0:
                self._planner_context = self._context_from_cache()
            self._run_planner(mode, move_dir, face_dir, target_vel, height)

        # 3. Sample 10 future frames from the cache at stride 5
        playback = self._playback_idx
        idx = (playback.view(-1, 1) + self._future_offsets.view(1, -1)).clamp(
            max=RESAMPLED_FRAMES - 1
        )  # [N, 10]
        idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
        n_range = (
            torch.arange(self.num_envs, device=self.device)
            .view(-1, 1)
            .expand(-1, G1_NUM_FUTURE_FRAMES)
        )
        frames = self._planner_cache[n_range, idx]  # [N, 10, 36]
        frames_next = self._planner_cache[n_range, idx_next]

        # 4. Lower-body future: legs from planner (MJ → IL via leg_mj_slots)
        joints_mj_future = frames[..., 7:]  # [N, 10, 29] MJ
        joints_mj_future_next = frames_next[..., 7:]
        leg_pos_mj = joints_mj_future[..., self._leg_mj_slots]  # [N, 10, 12]
        leg_vel_mj = (joints_mj_future_next[..., self._leg_mj_slots] - leg_pos_mj) * (
            1.0 / DT_POLICY
        )

        # 5. Assemble joint_pos_future [N, 10, 29] in IL body order
        joint_pos_future = torch.zeros(
            self.num_envs,
            G1_NUM_FUTURE_FRAMES,
            NUM_BODY_DOF,
            dtype=torch.float32,
            device=self.device,
        )
        joint_vel_future = torch.zeros_like(joint_pos_future)
        # Legs (IL scatter)
        joint_pos_future[..., self._leg_idx_il] = leg_pos_mj
        joint_vel_future[..., self._leg_idx_il] = leg_vel_mj
        # Upper: broadcast buffer across 10 frames; vel = 0
        joint_pos_future[..., self._upper_idx_il] = arm_targets_17.unsqueeze(1).expand(
            -1, G1_NUM_FUTURE_FRAMES, -1
        )

        # 6. ref_root_quat_future (world-frame wxyz from planner cache)
        ref_root_quat_future_wxyz = frames[..., 3:7]  # [N, 10, 4]

        # 7. Encoder / decoder
        target_il = self._inference.step(
            joint_pos_future=joint_pos_future,
            joint_vel_future=joint_vel_future,
            ref_root_quat_future_wxyz=ref_root_quat_future_wxyz,
            joint_pos=joint_pos_il,
            joint_vel=joint_vel_il,
            base_ang_vel=base_ang_vel,
            gravity_in_base=gravity_in_base,
            root_quat_wxyz=root_quat_wxyz,
        )  # [N, 29] IL

        # 8. Advance
        self._playback_idx = (self._playback_idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
        self._step_counter += 1

        return target_il

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset SonicG1Inference ring buffers + planner context/cache + step counter.

        Also zeroes the virtual `_facing_angle` (world +X), mirroring sonic pico
        `gamepad.hpp:621`: `planner_facing_angle = 0.0` when the planner is
        enabled / reset.
        """
        if not self._bound:
            return
        # Simplification: env-specific reset does a whole-policy reset (the
        # inference ring buffers are tensors and could be partially reset, but
        # the win is small). Matches HomiePolicy.reset style.
        self._inference.reset(
            joint_pos=self._default_jp_il.unsqueeze(0).expand(self.num_envs, -1)
        )
        self._seed_planner_context_init()
        self._facing_angle.zero_()
        self._step_counter = 0

    # ==================================================================
    # Accessors (consumed by SonicWBCAction)
    # ==================================================================
    @property
    def leg_idx_il(self) -> torch.Tensor:
        """Indices of the 12 leg joints within the 29-DOF body-IL array."""
        assert self._leg_idx_il is not None
        return self._leg_idx_il

    @property
    def body_idx_full(self) -> torch.Tensor:
        """Indices of the 29 body joints within the 43-DOF USD-full array (used by prepare_observations)."""
        assert self._body_idx_full is not None
        return self._body_idx_full

    @property
    def upper_default_il(self) -> torch.Tensor:
        """[17] default arm+waist angles used to seed SonicArmBuffer on reset."""
        assert self._upper_default_il is not None
        return self._upper_default_il

    @property
    def default_jp_il(self) -> torch.Tensor:
        """[29] body IL default joint pos。"""
        assert self._default_jp_il is not None
        return self._default_jp_il

    @property
    def is_bound(self) -> bool:
        return self._bound
