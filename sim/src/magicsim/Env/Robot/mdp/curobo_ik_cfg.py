"""Config for the unified :class:`CuroboIKAction` (cuRobo 2.0).

Single cfg class — ``action_dim = 7 * L`` where ``L = len(tool_frames)``
is declared in the robot YAML. Single-arm (``L=1``) and multi-arm
(``L>=2``, e.g. dual-arm) use the same class; the dual-arm path needs a
multi-tool-frame YAML declaring both EEFs. See
``CUROBO_V2_02_MIGRATION_PLAN.md`` §0.1 / §4 for the rationale.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import curobo_ik_actions


@configclass
class CuroboIKActionCfg(ActionTermCfg):
    """Configuration for the unified cuRobo 2.0 batched IK action term.

    Accepts a ``7 * L`` EEF pose target for every env — ``L`` consecutive
    7-tuples ``(x, y, z, qw, qx, qy, qz)`` in env-origin-relative world
    frame, ordered to match ``tool_frames``. All environments are solved
    in a single ``ik.solve_pose`` call. No scene/world collision is
    loaded; self-collision comes from the robot YAML.
    """

    class_type: type[ActionTerm] = curobo_ik_actions.CuroboIKAction

    joint_names: list[str] = MISSING
    """Regex patterns selecting the controlled joints across **all** tool frames.

    For a dual-arm robot with a multi-tool-frame YAML this spans both
    arms' joints (e.g. ``["R_joint[1-6]", "L_joint[1-6]"]``)."""

    robot_cfg_file: str = MISSING
    """cuRobo v2 robot YAML filename (resolved via ``get_robot_configs_path()``).

    The YAML must declare ``kinematics.tool_frames: [...]`` with length
    ``L >= 1``. ``L > 1`` transparently enables multi-tool-frame IK."""

    tool_frames: list[str] | None = None
    """Explicit list of tool frame names to track. Must be a subset of the
    YAML's ``kinematics.tool_frames`` (order matters — the ``7 * L`` action
    slices map to this order). If ``None`` (the default), the YAML's
    ``tool_frames`` is used verbatim, keeping single-arm cfgs minimal."""

    action_space: torch.Tensor = MISSING
    """``(2, 7 * L)`` tensor giving ``[low, high]`` for the concatenated
    ``L`` 7-vectors."""

    num_seeds: int = 8
    """Number of random seeds used by the cuRobo IK solver.

    Tuned for action-tracking IK (per-physics-step): each target is a
    small delta from the current joint state, so the default seed (=
    current pose) converges in a few LM iterations and extra random
    seeds rarely win. 8 is ~3× faster than 20 and converges on all
    tested robots. For planning-side IK (global argmin search across
    far-apart candidates), use IKServer's ``ik_num_seeds`` instead —
    that path needs the wider seed coverage."""

    position_threshold: float = 0.005
    """Convergence threshold for position (metres); maps to v2's ``position_tolerance``."""

    rotation_threshold: float = 0.05
    """Convergence threshold for rotation (radians); maps to v2's ``orientation_tolerance``."""

    self_collision_check: bool = True
    """Whether to check self-collision during IK.

    Must stay True when ``scene_model=None`` (the default for
    action-tracking IK). Turning it off with no scene leaves the IK
    solver with zero active costs and makes the MPPI sampler's
    ``torch.cat`` of per-cost tensors fail on an empty list
    (``metrics.py:259``). If you want to save the ~30% self-collision
    cost, add a no-op scene (``scene_model`` with a tiny dummy
    collision primitive) so the cost list stays non-empty."""

    fallback_to_current_on_fail: bool = True
    """If True, keep current joint positions for envs whose IK fails."""

    curobo_device: str | None = None
    """Device string for the cuRobo solver (``"cuda:0"``); None → pick the
    sim device if it is CUDA, else the default CUDA device. cuRobo 2.0
    requires CUDA."""

    decimation: int = 1
    """Run the full cuRobo solve every ``N`` ``process_actions`` calls.
    The first tick always fires cuRobo regardless so ``processed_actions``
    is never the zero-init. Intermediate ticks use diff-IK when
    ``diff_ik_method`` is set and ``L == 1``; otherwise the last cuRobo
    solution is held."""

    diff_ik_method: str | None = "pinv"
    """Jacobian inversion method for the inter-decimation diff-IK.
    Choices: ``'pinv'``, ``'svd'``, ``'trans'``, ``'dls'``, or ``None``
    to disable diff-IK refinement (hold the last cuRobo solution between
    ticks). For the unified :class:`CuroboIKAction`, only active when
    ``L == 1``. For :class:`DualCuroboIKAction`, runs per-arm at ``L == 2``."""

    world_to_base_frame: bool = False
    """When ``True``, the action term re-expresses each tick's input from
    world frame to the cuRobo solver's base-link frame before feeding the
    solver. The base link's live world pose comes from
    ``body_link_state_w[base_link_idx]`` (matching pink IK's
    ``_get_base_link_frame_transform`` path) — NOT from
    ``root_pos_w``, which on a mobile manipulator points at the
    articulation root (``vega_1p_mobile``), not the cuRobo base link
    (``vega_1p_base``) that sits below the dummy_base virtual joints.

    Set ``base_link_name`` to the link to use for the transform. If left
    ``None`` while this flag is True, the action falls back to
    ``root_pos_w`` and warns — only correct when the articulation root
    *is* the cuRobo base link.

    Default ``False`` keeps the original semantics: input is taken as
    env-origin-relative and passed directly to cuRobo (correct only when
    the robot lives at ``env_origin`` AND the cuRobo base link is the
    articulation root)."""

    base_link_name: str | None = None
    """Body name of the cuRobo solver's base link (matches the yaml's
    ``kinematics.base_link``). Used by the world→base transform to read
    the live link pose via ``body_link_state_w``. Required when
    ``world_to_base_frame`` is True and the cuRobo base link differs from
    the articulation root (e.g. vega's ``vega_1p_base`` sits below
    ``vega_1p_mobile`` + the dummy_base virtual joints)."""


@configclass
class DualCuroboIKActionCfg(CuroboIKActionCfg):
    """v2 dual-arm cfg — two kinematically-independent arms in one batched solve.

    cuRobo still runs as multi-tool-frame batched IK (``L = 2`` tool frames
    from a dual-arm YAML; single :class:`InverseKinematics` instance, all
    ``num_envs`` problems solved in parallel). What changes vs. the unified
    :class:`CuroboIKActionCfg` is the inter-decimation fallback:
    :class:`DualCuroboIKAction` runs **per-arm** differential-IK between
    cuRobo ticks, using each arm's own Jacobian slice + joint subset. Use
    this when ``decimation > 1`` and you want refinement between ticks on
    a two-separate-arms robot (DualSO101 / DualPiper / DualArxX5).

    Convention: right arm first, left arm second (matches v1). The unified
    ``joint_names`` / ``tool_frames`` fields inherited from
    :class:`CuroboIKActionCfg` are filled from the ``right_*`` / ``left_*``
    fields in :meth:`__post_init__` — don't set them by hand.
    """

    class_type: type[ActionTerm] = curobo_ik_actions.DualCuroboIKAction

    right_joint_names: list[str] = MISSING
    """Regex patterns selecting the right-arm joints (e.g. ``["R_joint[1-6]"]``)."""

    left_joint_names: list[str] = MISSING
    """Regex patterns selecting the left-arm joints."""

    right_eef_link_name: str = MISSING
    """Right-arm EEF body name. Must appear in the YAML's
    ``kinematics.tool_frames`` list."""

    left_eef_link_name: str = MISSING
    """Left-arm EEF body name. Must appear in the YAML's
    ``kinematics.tool_frames`` list."""

    # Unified fields inherited from CuroboIKActionCfg are auto-filled below.
    joint_names: list[str] = None  # type: ignore[assignment]
    tool_frames: list[str] | None = None

    def __post_init__(self) -> None:
        if self.right_joint_names is MISSING or self.left_joint_names is MISSING:
            raise ValueError(
                "DualCuroboIKActionCfg requires right_joint_names and left_joint_names."
            )
        if self.right_eef_link_name is MISSING or self.left_eef_link_name is MISSING:
            raise ValueError(
                "DualCuroboIKActionCfg requires right_eef_link_name and left_eef_link_name."
            )
        # Right first, left second — matches v1 dual convention. Hard-override
        # the unified fields even if someone passed them in (they are
        # derived, not authored).
        self.joint_names = list(self.right_joint_names) + list(self.left_joint_names)
        self.tool_frames = [self.right_eef_link_name, self.left_eef_link_name]
