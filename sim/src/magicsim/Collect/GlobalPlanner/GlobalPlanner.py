from typing import Any, Dict, Optional, Tuple, Union

import torch
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Planner.Utils import quat_normalize
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class GlobalPlanner:
    """
    Global Planner for all tasks.

    Subclasses share command header / flat target parsing via ``parse_planner_header``,
    ``eef_num_from_hand_id``, ``_eef_layout_from_robot``, and ``parse_target_vector``.

    Non-mobile manipulators (MoveL, ServoL, …) use ``_build_full_action_manipulator`` for the
    stacked ``base | arm | eef`` output (via each class's ``_build_full_action``).

    Mobile humanoids (``MobileMoveL``, ``RetractMoveL``, ``MobileServoL``, …) use
    ``_build_full_action_mobile``: base slice includes G1/PController ``lock_flag``, MotionGen
    ``base+arm`` row handling, and ``current_eef_target`` / ``_expand_eef_action``.
    """

    # Kept identical to legacy MobileMoveL / RetractMoveL error text.
    ERR_NO_BASE_IN_TARGET = "Mobile Curobo Control Base"

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.config = config
        self.env = env
        self.env_id = env_id
        self.logger = logger
        self.current_state = (
            None  # should be one of "ready", "running", "failed", "truncated"
        )
        self.current_command = None  # command that global planner get
        self.current_action = None  # action that global planner output
        self.step_count = 0

    def reset(self, action: torch.Tensor):
        raise NotImplementedError

    def step(self) -> torch.Tensor:
        raise NotImplementedError

    def update(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def parse_planner_header(
        action: Any,
        *,
        default_robot_id: int = 0,
        default_hand_id: int = 0,
    ) -> Tuple[int, int, int, Any]:
        """Parse ``((robot_id, hand_id, planner_mode), target)`` or legacy forms.

        The third integer is ``planner_mode`` (semantics depend on the concrete planner).
        Default when omitted is ``-1``.

        Legacy:
            - ``target_tensor`` only → ``default_robot_id``, ``default_hand_id``, third ``-1``
            - ``(robot_id, target_tensor)`` → ``hand_id=0``, third ``-1``
        """
        robot_id = default_robot_id
        hand_id = default_hand_id
        planner_mode = -1
        target = action
        if isinstance(action, (list, tuple)) and len(action) == 2:
            header = action[0]
            target = action[1]
            if isinstance(header, (list, tuple)):
                robot_id = int(header[0])
                hand_id = int(header[1]) if len(header) > 1 else 0
                planner_mode = int(header[2]) if len(header) > 2 else -1
            else:
                robot_id = int(header)
        return robot_id, hand_id, planner_mode, target

    @staticmethod
    def eef_num_from_hand_id(hand_id: int) -> int:
        if hand_id in (0, 1):
            return 1
        if hand_id == -1:
            return 2
        raise ValueError(f"hand_id must be 0, 1, or -1, got {hand_id}")

    def _eef_layout_from_robot(self) -> Tuple[int, int]:
        """``(max_eef_num, per_eef_dim)`` from ``RobotManager.get_info()`` (same as planner cfg)."""
        rn = getattr(self, "robot_name", None)
        if rn is None:
            raise RuntimeError(
                "GlobalPlanner: robot_name is not set; set robot before parsing target "
                "(e.g. _set_robot_by_id)."
            )
        scene = getattr(self.env, "scene", None)
        if scene is None:
            raise RuntimeError("GlobalPlanner: env.scene is missing.")
        rm = getattr(scene, "robot_manager", None)
        if rm is None:
            raise RuntimeError("GlobalPlanner: env.scene.robot_manager is missing.")
        info = rm.get_info()
        if rn not in info:
            raise RuntimeError(
                f"GlobalPlanner: robot_name {rn!r} not in RobotManager.get_info() "
                f"(keys: {list(info.keys())})."
            )
        ri = info[rn]
        if "max_eef_num" not in ri:
            raise RuntimeError(
                f"GlobalPlanner: RobotManager.get_info()[{rn!r}] missing max_eef_num."
            )
        if "per_eef_dim" not in ri:
            raise RuntimeError(
                f"GlobalPlanner: RobotManager.get_info()[{rn!r}] missing per_eef_dim."
            )
        return int(ri["max_eef_num"]), int(ri["per_eef_dim"])

    def _build_full_action_manipulator(
        self, arm_action_flat: torch.Tensor
    ) -> torch.Tensor:
        """Stack full planner action: optional base (NaN or ``current_base_target``), arm, optional eef.

        For fixed-base / non-mobile arms (MoveL, ServoL, …). Requires ``_get_action_dims``,
        ``_expand_arm_action``, ``_expand_eef_target``, and ``current_*_target`` attributes.
        """
        base_dim, arm_dim, eef_dim = self._get_action_dims()

        arm_action_flat = arm_action_flat.view(-1)
        if arm_action_flat.shape[0] != arm_dim:
            arm_action_flat = self._expand_arm_action(arm_action_flat)
        if arm_action_flat.shape[0] != arm_dim:
            raise ValueError(
                f"arm_action length {arm_action_flat.shape[0]} does not match arm_dim {arm_dim}."
            )

        chunks = []
        if base_dim > 0:
            if self.current_base_target is None:
                chunks.append(
                    torch.tensor(
                        [torch.nan] * base_dim,
                        device=arm_action_flat.device,
                        dtype=arm_action_flat.dtype,
                    )
                )
            else:
                chunks.append(self.current_base_target)
        chunks.append(arm_action_flat)
        if eef_dim > 0:
            if self.current_eef_target is None:
                chunks.append(
                    torch.tensor(
                        [torch.nan] * eef_dim,
                        device=arm_action_flat.device,
                        dtype=arm_action_flat.dtype,
                    )
                )
            else:
                chunks.append(self._expand_eef_target(self.current_eef_target))
        output_action = torch.cat(chunks, dim=0)
        assert output_action.shape[0] == base_dim + arm_dim + eef_dim, (
            f"Output action shape {output_action.shape[0]} does not match expected "
            f"base_dim={base_dim}, arm_dim={arm_dim}, eef_dim={eef_dim}."
        )
        return output_action

    def _build_full_action_mobile(
        self,
        arm_action_flat: torch.Tensor,
        lock_flag_override: Optional[float] = None,
    ) -> torch.Tensor:
        """Full robot action for mobile base + arm (+ eef): ``base | arm | eef``.

        ``base_dim`` is planner base action width (includes ``lock_flag`` as last scalar).
        ``motiongen_base_dim = base_dim - 1`` is the pose part MotionGen outputs before the flag.

        Requires: ``_get_action_dims``, ``_eef_layout_from_robot`` (for ``max_eef_num``), ``_expand_arm_action``,
        ``_expand_eef_action``, ``_use_locked_base``, ``current_eef_target``.
        """
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        motiongen_base_dim = base_dim - 1 if base_dim > 0 else 0
        if lock_flag_override is not None:
            lock_flag_val = float(lock_flag_override)
        elif getattr(self, "_use_locked_base", False):
            lock_flag_val = -1.0
        else:
            lock_flag_val = 0.0

        if getattr(self, "robot_name", None) is None:
            motiongen_input_arm_dims = (7,)
        else:
            max_eef_num, _ = self._eef_layout_from_robot()
            dims_set = {max_eef_num * 7}
            if max_eef_num >= 2:
                dims_set.add(7)
            motiongen_input_arm_dims = tuple(sorted(dims_set))

        arm_action_flat = arm_action_flat.view(-1)
        if base_dim > 0 and arm_action_flat.shape[0] == motiongen_base_dim + arm_dim:
            base_pose = arm_action_flat[:motiongen_base_dim]
            arm_action = self._expand_arm_action(arm_action_flat[motiongen_base_dim:])
            base_action = torch.cat(
                [
                    base_pose,
                    torch.tensor(
                        [lock_flag_val],
                        device=arm_action_flat.device,
                        dtype=arm_action_flat.dtype,
                    ),
                ]
            )
        elif arm_action_flat.shape[0] == arm_dim:
            arm_action = self._expand_arm_action(arm_action_flat)
            if base_dim > 0:
                base_action = torch.cat(
                    [
                        torch.full(
                            (motiongen_base_dim,),
                            torch.nan,
                            device=arm_action_flat.device,
                            dtype=arm_action_flat.dtype,
                        ),
                        torch.tensor(
                            [lock_flag_val],
                            device=arm_action_flat.device,
                            dtype=arm_action_flat.dtype,
                        ),
                    ]
                )
            else:
                base_action = None
        elif arm_action_flat.shape[0] in motiongen_input_arm_dims:
            arm_action = self._expand_arm_action(arm_action_flat)
            base_action = None
        else:
            raise ValueError(
                f"arm_action length {arm_action_flat.shape[0]} does not match "
                f"arm_dim {arm_dim} or arm_dim+motiongen_base_dim {arm_dim + motiongen_base_dim}."
            )

        chunks = []
        if base_dim > 0:
            if base_action is None:
                base_action = torch.full(
                    (base_dim,),
                    torch.nan,
                    device=arm_action_flat.device,
                    dtype=arm_action_flat.dtype,
                )
            chunks.append(base_action)
        chunks.append(arm_action)
        if eef_dim > 0:
            if self.current_eef_target is None:
                chunks.append(
                    torch.tensor(
                        [torch.nan] * eef_dim,
                        device=arm_action_flat.device,
                        dtype=arm_action_flat.dtype,
                    )
                )
            else:
                chunks.append(
                    self._expand_eef_action(self.current_eef_target).to(
                        device=arm_action_flat.device, dtype=arm_action_flat.dtype
                    )
                )
        output_action = torch.cat(chunks, dim=0)
        expected_dim = arm_dim + base_dim + eef_dim
        assert output_action.shape[0] == expected_dim, (
            f"Output action shape {output_action.shape[0]} does not match expected "
            f"arm_dim={arm_dim}, base_dim={base_dim}, expected total {expected_dim}."
        )
        return output_action

    @staticmethod
    def parse_target_vector(
        action: Union[torch.Tensor, Any],
        *,
        device: torch.device,
        base_dim: int,
        eef_dim: int,
        hand_id: int,
        allow_base: bool,
        max_eef_num: int,
        per_eef_dim: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Parse flat target: arm poses (``active_eef_num * 7``), optional per-eef dex, optional base.

        ``active_eef_num`` from ``hand_id``: 1 for 0/1, 2 for ``-1`` (both arms). It must be
        ``<= max_eef_num``.

        ``max_eef_num`` and ``per_eef_dim`` must match ``RobotManager.get_info()`` for this robot
        (``per_eef_dim * max_eef_num`` must equal ``eef_dim`` when ``eef_dim > 0``).

        * **MoveL** (``allow_base=True``): ``arm``, ``arm+eef``, or ``base+arm`` / ``base+arm+eef``.
        * **MobileMoveL / RetractMoveL** (``allow_base=False``): ``arm`` or ``arm+eef`` only; if a
          layout that includes ``base_dim`` in the vector is detected, raises
          ``GlobalPlanner.ERR_NO_BASE_IN_TARGET``.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=device, dtype=torch.float32)
        else:
            action = action.to(device, dtype=torch.float32)
        if action.ndim != 1:
            action = action.view(-1)

        active_eef_num = GlobalPlanner.eef_num_from_hand_id(hand_id)
        if active_eef_num > max_eef_num:
            raise ValueError(
                f"hand_id={hand_id} implies {active_eef_num} active EEFs but "
                f"max_eef_num is {max_eef_num}."
            )
        expected_arm_dim = active_eef_num * 7
        if eef_dim > 0:
            if per_eef_dim * max_eef_num != eef_dim:
                raise ValueError(
                    f"eef layout mismatch: eef_dim={eef_dim}, max_eef_num={max_eef_num}, "
                    f"per_eef_dim={per_eef_dim} (require per_eef_dim * max_eef_num == eef_dim)."
                )
        elif per_eef_dim != 0:
            raise ValueError(
                f"eef_dim is 0 but per_eef_dim={per_eef_dim} (expected 0 from RobotManager.get_info)."
            )
        matched_eef_dim = per_eef_dim * active_eef_num

        total_dim = int(action.shape[0])

        if not allow_base and base_dim > 0:
            if total_dim in (
                base_dim + expected_arm_dim,
                base_dim + expected_arm_dim + matched_eef_dim,
            ):
                raise ValueError(GlobalPlanner.ERR_NO_BASE_IN_TARGET)

        base_target = None
        eef_target = None
        arm_target_flat: Optional[torch.Tensor] = None

        if allow_base and base_dim > 0:
            if total_dim == expected_arm_dim:
                arm_target_flat = action.clone()
            elif (
                matched_eef_dim > 0 and total_dim == expected_arm_dim + matched_eef_dim
            ):
                arm_target_flat = action[:expected_arm_dim].clone()
                eef_target = action[expected_arm_dim:].clone()
            elif total_dim == base_dim + expected_arm_dim:
                base_target = action[:base_dim].clone()
                arm_target_flat = action[base_dim : base_dim + expected_arm_dim].clone()
            elif (
                matched_eef_dim > 0
                and total_dim == base_dim + expected_arm_dim + matched_eef_dim
            ):
                base_target = action[:base_dim].clone()
                arm_target_flat = action[base_dim : base_dim + expected_arm_dim].clone()
                eef_target = action[base_dim + expected_arm_dim :].clone()
        else:
            if total_dim == expected_arm_dim:
                arm_target_flat = action.clone()
            elif (
                matched_eef_dim > 0 and total_dim == expected_arm_dim + matched_eef_dim
            ):
                arm_target_flat = action[:expected_arm_dim].clone()
                eef_target = action[expected_arm_dim:].clone()

        if arm_target_flat is None:
            valid = [str(expected_arm_dim)]
            if matched_eef_dim > 0:
                valid.append(f"{expected_arm_dim}+{matched_eef_dim}")
            if allow_base and base_dim > 0:
                valid.append(f"base({base_dim})+{expected_arm_dim}")
                if matched_eef_dim > 0:
                    valid.append(
                        f"base({base_dim})+{expected_arm_dim}+{matched_eef_dim}"
                    )
            raise ValueError(
                f"target length {total_dim} does not match allowed layouts "
                f"({', '.join(valid)}); hand_id={hand_id}, active_eef_num={active_eef_num}."
            )

        arm_targets = arm_target_flat.view(active_eef_num, 7)
        arm_targets = torch.cat(
            [arm_targets[:, :3], quat_normalize(arm_targets[:, 3:])],
            dim=1,
        )

        if eef_target is not None and per_eef_dim > 0:
            eef_target = eef_target.view(active_eef_num, per_eef_dim)

        return base_target, arm_targets, eef_target
