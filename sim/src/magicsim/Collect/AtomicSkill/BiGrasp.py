"""BiGrasp atomic skill: paired bimanual parallel-gripper grasp.

Mirror of :class:`magicsim.Collect.AtomicSkill.Grasp.Grasp` but drives BOTH
arms synchronously through every phase (the single-arm version sequenced
right-then-left, which lifted the basket with one arm before the other
arm could approach — useless for a bimanual carry).

Phases — both arms move in lock-step:

    pre_grasp     : both eefs at grasp_pose - 0.15 m along approach axis,
                    grippers open.
    grasp         : both eefs at the paired grasp pose, grippers open.
    close_gripper : ParallelGripper(both) → closed.
    retrieval     : both eefs at grasp_pose + 0.30 m along world +z,
                    grippers closed (this is what actually lifts the
                    basket bimanually).

IK is called ONCE in :meth:`reset` to pick a paired candidate from the
basket's ``bi_gripper_grasp_pose`` annotation; subsequent phase poses are
deterministic offsets of that single chosen pair, just like single-arm
:class:`Grasp` does. Re-running IK each phase is unnecessary and would
risk picking different pairs across phases (= losing grip mid-lift).

Action format (matches DualFranka frame order: right first, left second
— see :file:`Env/Robot/Cfg/DualManipulator/DualFranka.py:10`)::

    16D = [right_arm_pose(7), left_arm_pose(7), right_grip(1), left_grip(1)]
"""

from typing import Any, List

import concurrent.futures
import torch
from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Planner.PlannerManager import PlannerManager
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Task.LocoManip.Env.Test.TestLocoGraspEnv import visualize_grasp_pose


class BiGrasp(AtomicSkill):
    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = 0
        self.hand_id = -1  # bimanual

        self.pre_grasp_offset = float(getattr(config, "pre_grasp_offset", 0.15))
        self.retrieval_offset = float(getattr(config, "retrieval_offset", 0.30))
        self.viz_grasp = bool(getattr(config, "viz_grasp", True))
        self._last_viz_phase: str | None = None
        self.functional_grasp = True
        self.part = None
        self.obj_type: str | None = None
        self.obj_name: str | None = None
        self.obj_id: int | None = None

        self.current_phase: str | None = None
        self.r_grasp: torch.Tensor | None = None
        self.l_grasp: torch.Tensor | None = None
        self.r_pre: torch.Tensor | None = None
        self.l_pre: torch.Tensor | None = None
        self.r_retrieval: torch.Tensor | None = None
        self.l_retrieval: torch.Tensor | None = None

        self.robot_name: str | None = None
        self.ik_server = None
        self.planner_manager: PlannerManager | None = None

        self._ik_job: dict | None = None
        self._ik_token: int = 0

    # ------------------------------------------------------------------ helpers
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

    def _shift_along_grasp_dir(
        self, pose7: torch.Tensor, offset: float, backward: bool = True
    ) -> torch.Tensor:
        """Move along the gripper's local +z (approach) axis by ``offset``."""
        from magicsim.Env.Utils.rotations import quat_to_rot_matrix

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

    # Phase target poses are stored in env-local frame (basket pose ⊕
    # local annotation). Isaac debug-draw expects world coords, so we add
    # the env origin before drawing.
    def _to_world_for_viz(self, pose7: torch.Tensor) -> torch.Tensor:
        device = self.env.device
        pose = pose7.to(device).clone()
        origin = self.env.scene.env_origins[self.env_id].to(device)
        if origin.ndim > 1:
            origin = origin[0]
        pose[:3] = pose[:3] + origin[:3]
        return pose

    def _viz_for_phase(self, phase: str | None) -> None:
        if not self.viz_grasp or phase is None:
            return
        if phase == "pre_grasp":
            right_pose, left_pose = self.r_pre, self.l_pre
        elif phase in ("grasp", "close_gripper"):
            right_pose, left_pose = self.r_grasp, self.l_grasp
        elif phase == "retrieval":
            right_pose, left_pose = self.r_retrieval, self.l_retrieval
        else:
            return
        if right_pose is None or left_pose is None:
            return
        visualize_grasp_pose(
            [self._to_world_for_viz(right_pose), self._to_world_for_viz(left_pose)]
        )

    # ------------------------------------------------------------------ IK
    def _submit_paired_goalset(
        self, rights: torch.Tensor, lefts: torch.Tensor
    ) -> concurrent.futures.Future:
        rs = self._get_robot_state()
        rs_dict = {
            "base_pos": rs["base_pos"],
            "base_quat": rs["base_quat"],
            "joint_pos": rs["joint_pos"],
            "joint_vel": rs["joint_vel"],
        }
        # (1, G, 14): slot 0:7 = right, slot 7:14 = left.
        G = rights.shape[0]
        target = torch.empty((1, G, 14), device=self.env.device, dtype=torch.float32)
        target[0, :, :7] = rights.to(self.env.device)
        target[0, :, 7:] = lefts.to(self.env.device)

        is_dual = bool(getattr(self.ik_server, "dual_mode", False))
        if is_dual:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=rs_dict,
                mode="goalset",
                lock_base=False,
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=rs_dict,
                mode="goalset",
            )
        return self.ik_server.submit_ik(req)

    def _start_ik_job(self, pairs: List[dict]):
        rights = torch.stack([p["right"] for p in pairs], dim=0).to(self.env.device)
        lefts = torch.stack([p["left"] for p in pairs], dim=0).to(self.env.device)
        self._ik_token += 1
        self._ik_job = {
            "token": self._ik_token,
            "rights": rights,
            "lefts": lefts,
            "future": self._submit_paired_goalset(rights, lefts),
        }

    def _poll_ik_job(self) -> bool:
        """Return True when poses are ready; False while computing; raises if failed."""
        if self.r_grasp is not None:
            return True
        if self._ik_job is None:
            return False
        fut: concurrent.futures.Future = self._ik_job["future"]
        if not fut.done():
            self.current_state = "computing"
            return False
        try:
            success_list, idx_list, ret_envs = fut.result()
        except Exception as ex:
            print(f"[BiGrasp] paired IK exception: {ex}")
            self.current_state = f"failed: ik exception {ex}"
            self._ik_job = None
            return False
        ok = bool(success_list[0]) if success_list else False
        idx = int(idx_list[0]) if idx_list else -1
        if not ok or idx < 0:
            print(f"[BiGrasp] paired IK failed: success={success_list} idx={idx_list}")
            self.current_state = "failed: paired ik no solution"
            self._ik_job = None
            return False
        rights = self._ik_job["rights"]
        lefts = self._ik_job["lefts"]
        self.r_grasp = rights[idx].clone()
        self.l_grasp = lefts[idx].clone()
        self.r_pre = self._shift_along_grasp_dir(
            self.r_grasp, self.pre_grasp_offset, backward=True
        )
        self.l_pre = self._shift_along_grasp_dir(
            self.l_grasp, self.pre_grasp_offset, backward=True
        )
        self.r_retrieval = self._shift_world_z(self.r_grasp, self.retrieval_offset)
        self.l_retrieval = self._shift_world_z(self.l_grasp, self.retrieval_offset)
        rg = self.r_grasp[:3].cpu().tolist()
        lg = self.l_grasp[:3].cpu().tolist()
        print(
            f"[BiGrasp] env_id={self.env_id} chose paired idx={idx}/{rights.shape[0]}: "
            f"R=({rg[0]:+.2f},{rg[1]:+.2f},{rg[2]:+.2f}) "
            f"L=({lg[0]:+.2f},{lg[1]:+.2f},{lg[2]:+.2f})"
        )
        self._ik_job = None
        return True

    # ------------------------------------------------------------------ reset / refresh
    def reset(self, action: List[Any]):
        # action: ["BiGrasp", robot_id, obj_type, obj_name, obj_id, functional_grasp, part]
        self.robot_id = int(action[1])
        self.hand_id = -1
        self.obj_type = action[2]
        self.obj_name = action[3]
        self.obj_id = action[4]
        self.functional_grasp = bool(action[5]) if len(action) > 5 else True
        self.part = action[6] if len(action) > 6 else None
        self.current_command = list(action)
        self.current_state = "ready"
        self.current_phase = "pre_grasp"
        self.r_grasp = self.l_grasp = None
        self.r_pre = self.l_pre = None
        self.r_retrieval = self.l_retrieval = None
        self._ik_job = None
        self._last_viz_phase = None

        self.planner_manager = self._get_planner_manager()
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            raise RuntimeError("IKServer not available in PlannerManager.")
        name = self._resolve_robot_name()
        if name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{name}'.")
        self.ik_server = ik_dict[name]

        if not hasattr(self.env, "get_bigripper_pairs_flat"):
            self.current_state = "failed: env lacks get_bigripper_pairs_flat"
            return
        pairs = self.env.get_bigripper_pairs_flat(
            env_id=self.env_id,
            obj_name=self.obj_name,
            obj_id=int(self.obj_id),
            functional_grasp=self.functional_grasp,
            part=self.part,
        )
        if not pairs:
            self.current_state = "failed: no bigripper grasp pairs"
            return

        # DEBUG: dump basket world pose + first/last target so we can spot
        # stale-state issues across trajectories (basket landed elsewhere
        # after a failed lift, robot joints not actually back to initial,
        # etc.). One line per reset, light overhead.
        try:
            obj_poses = self.env.get_object_pose(env_ids=[self.env_id])
            bp = obj_poses.get(self.obj_name, None)
            if bp is not None:
                bp = bp[0].cpu().tolist()
                print(
                    f"[BiGrasp][reset] env_id={self.env_id} "
                    f"basket world=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f}) "
                    f"quat=({bp[3]:+.3f},{bp[4]:+.3f},{bp[5]:+.3f},{bp[6]:+.3f}) "
                    f"n_pairs={len(pairs)}"
                )
            rs = self._get_robot_state()
            jp = rs["joint_pos"]
            jp0 = jp[self.env_id].cpu().tolist() if jp.dim() > 1 else jp.cpu().tolist()
            jp_fmt = ",".join(f"{v:+.2f}" for v in jp0[:8])
            print(f"[BiGrasp][reset] env_id={self.env_id} joint_pos[:8]=[{jp_fmt}] ...")
            r0 = pairs[0]["right"].cpu().tolist()
            l0 = pairs[0]["left"].cpu().tolist()
            print(
                f"[BiGrasp][reset] env_id={self.env_id} "
                f"pair[0] R=({r0[0]:+.3f},{r0[1]:+.3f},{r0[2]:+.3f}) "
                f"L=({l0[0]:+.3f},{l0[1]:+.3f},{l0[2]:+.3f})"
            )
        except Exception as ex:
            print(f"[BiGrasp][reset] debug dump failed: {ex}")

        self._start_ik_job(pairs)

    def refresh(self, action: List[Any]):
        new_cmd = list(action)
        if self.current_command != new_cmd or self.current_phase is None:
            self.reset(action)

    # ------------------------------------------------------------------ step / update
    def _build_dual_action(
        self,
        right_pose: torch.Tensor,
        left_pose: torch.Tensor,
        grip: float,
    ) -> torch.Tensor:
        """16D bimanual MoveL target: ``[r_pose(7), l_pose(7), r_grip(1), l_grip(1)]``."""
        device = self.env.device
        gripper = torch.tensor([grip, grip], device=device, dtype=torch.float32)
        return torch.cat([right_pose.to(device), left_pose.to(device), gripper], dim=0)

    def step(self):
        if self.current_state and self.current_state.startswith("failed"):
            self.current_action = "Failed"
            return "Failed"

        if not self._poll_ik_job():
            # If the poll TRANSITIONED to failed on this very call, report
            # the failure NOW — otherwise step() returns None this frame,
            # update() sees ``truncated=4`` and clears the atomic skill,
            # and the next step gets a brand-new skill before the env
            # ever sees ``failed_env_ids`` and runs reset_idx. Net effect
            # is "atomic skill keeps respawning into the same broken
            # state" without an env reset between attempts.
            if self.current_state and self.current_state.startswith("failed"):
                self.current_action = "Failed"
                return "Failed"
            self.current_action = None
            return None

        self.current_state = "running"
        if self.current_phase != self._last_viz_phase:
            self._viz_for_phase(self.current_phase)
            self._last_viz_phase = self.current_phase

        if self.current_phase == "close_gripper":
            # Hand off to ParallelGripper with hand_id=-1 — closes both
            # grippers without re-planning arm motion. MoveL would also
            # work but treats unchanged arm-pose targets as already-done
            # and might transition before the grippers fully close.
            gripper_target = torch.tensor(
                [1.0, 1.0], device=self.env.device, dtype=torch.float32
            )
            self.current_action = {
                "ParallelGripper": ((self.robot_id, -1, 0), gripper_target)
            }
            return self.current_action

        # MoveL phases (open during pre/grasp, closed during retrieval).
        grip = 1.0 if self.current_phase == "retrieval" else 0.0
        if self.current_phase == "pre_grasp":
            right_pose, left_pose = self.r_pre, self.l_pre
        elif self.current_phase == "grasp":
            right_pose, left_pose = self.r_grasp, self.l_grasp
        elif self.current_phase == "retrieval":
            right_pose, left_pose = self.r_retrieval, self.l_retrieval
        else:
            self.current_state = "failed: bad phase"
            self.current_action = "Failed"
            return "Failed"

        target_16d = self._build_dual_action(right_pose, left_pose, grip)
        # hand_id = -1 → MoveL routes the 14D pose to both arms (right 7
        # + left 7) and the 2D eef to both grippers.
        self.current_action = {"MoveL": ((self.robot_id, -1, -1), target_16d)}
        return self.current_action

    def update(self, info):
        base = {
            "atomic_skill_type": "BiGrasp",
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
            order = ["pre_grasp", "grasp", "close_gripper", "retrieval"]
            try:
                i = order.index(self.current_phase)
            except ValueError:
                i = -1
            if 0 <= i < len(order) - 1:
                self.current_phase = order[i + 1]
                print(f"[BiGrasp] env_id={self.env_id} phase={self.current_phase}")
                return {
                    **base,
                    "finished": False,
                    "state": f"running: {self.current_phase}",
                    "truncated": 0,
                    "phase": self.current_phase,
                }
            # retrieval finished
            self.current_state = "finished"
            return {
                **base,
                "finished": True,
                "state": "finished",
                "truncated": 0,
                "phase": "completed",
            }

        return {**base, "finished": False, "state": "running", "truncated": 0}
