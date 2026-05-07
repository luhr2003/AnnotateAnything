"""NaN / per-frame preprocessing + scene debug helpers for IK / MotionGen Servers.

Caller convention (used by AtomicSkill / GlobalPlanner):
- A 7-vec row of all-NaN xyz means **don't drive this (env, tool) on this
  candidate**. The Server transparently:
    1. Marks `(env, tool)` as disabled when EVERY candidate is NaN.
    2. For non-paired goalset: pads shorter tools to ``G`` by mod-cycling
       the real (non-NaN) candidates.
    3. For paired goalset: requires the per-env real_count to be EQUAL
       across active tools (else raises) — paired argmin demands jointly
       reachable candidate ``g``.
- The Server feeds curobo a non-NaN goal tensor; the per-env disable
  flips the tool-pose cost weight to zero on the disabled rows so the
  substituted value is semantically inert.

These helpers are pure functions: input tensor → mask + padded tensor.
They never mutate global state and never call into curobo. The Server
combines this with ``InverseKinematics.update_tool_pose_criteria_per_env``
to flip the per-env cost weights at runtime, on every solve.

Shapes
------
Single-pose:  ``batch_target_pos`` is ``(B, L_tracked, 7)``.
Goalset:      ``batch_target_pos`` is ``(B, L, G, 7)``.

NaN sentinel: ``batch_target_pos[..., :3].isnan().any(dim=-1)`` —
xyz-NaN is the canonical caller signal. Quaternion-NaN tags along but is
not the discriminator.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def log_scene_slot_load(
    scene,
    slot: int,
    env_id: int,
    tag: str = "",
) -> None:
    """Print one line per obstacle (mesh / cuboid / sphere / cylinder) in the
    given ``Scene`` as it is loaded into ``env_idx=slot``.

    Called by the IK / MotionGen Services right AFTER
    ``scene_collision_checker.load_collision_model(scene, env_idx=slot)``.
    Gated by the Server's ``self._debug`` flag at the call site so the
    print only fires when debug is on.

    Args:
        scene: :class:`curobo.scene.Scene` (a.k.a. ``SceneCfg``) or any
            object with ``.mesh``, ``.cuboid``, ``.sphere``, ``.cylinder``
            attributes. ``None`` / empty-scene inputs print a single
            "empty" line.
        slot: solver-local batch slot index (``env_idx=`` used in the
            ``load_collision_model`` call).
        env_id: caller-space game env id (``batch_env_ids[slot]``).
        tag: short label to disambiguate multiple callers in the log
            (e.g. ``"IK"``, ``"MG-locked"``, ``"MG-free"``, ``"pad"``).
    """
    prefix = f"[scene-load]{f'[{tag}]' if tag else ''} slot={slot} env_id={env_id}"
    if scene is None:
        print(f"{prefix} <None>")
        return
    groups = (
        ("mesh", getattr(scene, "mesh", None) or []),
        ("cuboid", getattr(scene, "cuboid", None) or []),
        ("sphere", getattr(scene, "sphere", None) or []),
        ("cylinder", getattr(scene, "cylinder", None) or []),
        ("capsule", getattr(scene, "capsule", None) or []),
    )
    total = sum(len(items) for _, items in groups)
    if total == 0:
        print(f"{prefix} <empty>")
        return
    print(f"{prefix} total={total}")
    for kind, items in groups:
        for obs in items:
            name = getattr(obs, "name", "?")
            pose = getattr(obs, "pose", None)
            if pose is not None and len(pose) >= 3:
                pos = (float(pose[0]), float(pose[1]), float(pose[2]))
                pos_str = f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            else:
                pos_str = "pos=<n/a>"
            print(f"{prefix}   {kind} '{name}' {pos_str}")


def log_joint_state_submit(
    current_state,
    batch_env_ids: List[int],
    tag: str = "",
) -> None:
    """Print per-slot ``JointState.position`` right before ``solve_pose`` /
    ``plan_pose``. Companion to :func:`log_scene_slot_load` for the
    debug stream.

    Args:
        current_state: the :class:`JointState` being submitted (expected
            shape ``(B, dof)`` on ``.position``). ``.joint_names`` is
            printed once at the top so each slot's numeric row stays
            compact.
        batch_env_ids: caller-space env ids, length ``B`` — slot ``b``
            maps to ``batch_env_ids[b]``.
        tag: short label to disambiguate callers (e.g. ``"IK"``,
            ``"MG-locked"``).
    """
    prefix = f"[js-submit]{f'[{tag}]' if tag else ''}"
    if current_state is None:
        print(f"{prefix} <None>")
        return
    pos = getattr(current_state, "position", None)
    joint_names = getattr(current_state, "joint_names", None)
    if pos is None:
        print(f"{prefix} <no .position>")
        return
    # Normalise (B, dof) — some paths carry (1, B, dof) or (B, 1, dof).
    if pos.dim() == 3 and pos.shape[0] == 1:
        pos = pos.squeeze(0)
    elif pos.dim() == 3 and pos.shape[1] == 1:
        pos = pos.squeeze(1)
    if pos.dim() != 2:
        print(f"{prefix} unexpected position shape {tuple(pos.shape)}")
        return
    B, dof = pos.shape
    names_str = (
        "[" + ", ".join(list(joint_names)) + "]"
        if joint_names is not None
        else "<unknown>"
    )
    print(f"{prefix} B={B} dof={dof} joint_names={names_str}")
    pos_cpu = pos.detach().to("cpu")
    for b in range(B):
        env_id = int(batch_env_ids[b]) if b < len(batch_env_ids) else -1
        row = pos_cpu[b].tolist()
        # 6 fractional digits — 4 hides physics-drift residuals (1e-4 to
        # 1e-3 mid-step) that often explain "current_state looked zero
        # but solver disagrees" debugging puzzles.
        row_str = "[" + ", ".join(f"{v:+.6f}" for v in row) + "]"
        print(f"{prefix}   slot={b} env_id={env_id} position={row_str}")


def detect_nan_disable_single(batch_target_pos: torch.Tensor) -> torch.Tensor:
    """Per (env, tool) NaN-row detection for the single-pose path.

    Args:
        batch_target_pos: shape ``(B, L_tracked, 7)``.

    Returns:
        Bool tensor of shape ``(B, L_tracked)`` where ``True`` means
        "this (env, tool) is NaN-masked → disable for this solve".
    """
    if batch_target_pos.ndim != 3 or batch_target_pos.shape[-1] != 7:
        raise ValueError(
            f"detect_nan_disable_single expects (B, L_tracked, 7); "
            f"got {tuple(batch_target_pos.shape)}"
        )
    return batch_target_pos[..., :3].isnan().any(dim=-1)


def detect_nan_and_pad_goalset(
    batch_target_pos: torch.Tensor,
    tracked_frames: List[str],
    paired: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per (env, tool) NaN classify + padding for the goalset path.

    Args:
        batch_target_pos: shape ``(B, L, G, 7)``. ``L`` MUST equal
            ``len(tracked_frames)``.
        tracked_frames: ordered list of frame names, length L (used only
            in error messages; the function does not look anything up).
        paired: when True, raises if per-env active-tool real_counts
            differ; when False, mod-pads shorter tools to ``G``.

    Returns:
        Tuple of:
        - ``padded`` (``(B, L, G, 7)``): same as input but with NaN slots
          inside partially-NaN tools replaced by mod-cycled real
          candidates. Tools whose every slot is NaN are returned
          unchanged — caller substitutes FK filler before building the
          GoalToolPose.
        - ``fully_nan_mask`` (``(B, L)`` bool): True for (env, tool)
          where every candidate is NaN.
        - ``real_count`` (``(B, L)`` int): per (env, tool) count of
          non-NaN candidates (0 .. G).

    Raises:
        ValueError: shape mismatch or paired equal-count violation.
    """
    if batch_target_pos.ndim != 4 or batch_target_pos.shape[-1] != 7:
        raise ValueError(
            f"detect_nan_and_pad_goalset expects (B, L, G, 7); "
            f"got {tuple(batch_target_pos.shape)}"
        )
    B, L, G, _ = batch_target_pos.shape
    if L != len(tracked_frames):
        raise ValueError(
            f"goalset target shape (B={B}, L={L}, G={G}, 7) does not match "
            f"tracked_frames length {len(tracked_frames)} ({tracked_frames})"
        )

    nan_xyz = batch_target_pos[..., :3].isnan().any(dim=-1)  # (B, L, G)
    real_count = (~nan_xyz).sum(dim=-1)  # (B, L) int
    fully_nan = real_count == 0  # (B, L) bool

    if paired:
        # For every env: among the tools that aren't fully NaN, real_count
        # must be the same. (One active tool per env is fine — paired
        # silently degrades to unpaired.)
        for b in range(B):
            active_li = (~fully_nan[b]).nonzero(as_tuple=True)[0]
            if active_li.numel() < 2:
                continue
            cnts = real_count[b, active_li].tolist()
            if len(set(cnts)) > 1:
                counts_by_frame = {
                    tracked_frames[int(li)]: int(c)
                    for li, c in zip(active_li.tolist(), cnts)
                }
                raise ValueError(
                    f"paired goalset: env {b} active tools have unequal "
                    f"non-NaN counts {counts_by_frame} — paired requires "
                    f"equal real_count across active tools (caller padding "
                    f"required, or set planner.ik.paired: false)."
                )

    # Mod-padding — only for partially-NaN tools. L and G are tiny in
    # practice (L ≤ 3 dual-arm + extras, G ≤ 32 grasp candidates) so the
    # explicit Python loop is fine and avoids gather-tensor allocation.
    out = batch_target_pos.clone()
    for b in range(B):
        for li in range(L):
            if fully_nan[b, li]:
                continue
            mask_g = ~nan_xyz[b, li]
            real_idx = mask_g.nonzero(as_tuple=True)[0]
            rc = int(real_idx.numel())
            if rc == G:
                continue
            for g in range(G):
                if not bool(mask_g[g]):
                    out[b, li, g] = batch_target_pos[b, li, real_idx[g % rc]]
    return out, fully_nan, real_count
