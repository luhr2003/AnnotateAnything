"""DexOpenDrawerEnv — open-drawer task driven by a Franka + Xhand dexterous hand.

Mirrors :class:`OpenDrawerEnv` but consumes 19-D actions
(``[x, y, z, qw, qx, qy, qz, j0..j11]``) so the 12-DOF Xhand fingers can be
commanded along the pre-recorded ``xhand_open_by_handle_trajectory``
annotation. Source data: ``~/sharpa_bin+xhand_open_by_handle/xhand_open_by_handle``.

The xhand annotation has a different shape than the parallel-gripper one:
each leaf is ``{"approach": [[19], …], "trajectory": [[19], …]}`` rather than
a flat ``[[7], …]`` list, so :meth:`Articulation.get_trajectory_poses` cannot
parse it. :meth:`get_drawer_trajectories` below loads the raw annotation
directly and concatenates ``approach + trajectory`` per ``(joint, traj_id)``.
"""

from typing import Any, Dict, Sequence

import torch

from magicsim.Task.TableTop.Env.OpenDrawerEnv import OpenDrawerEnv

# Franka arm pose (7) + Xhand finger joints (12) = 19 action dims.
ARM_POSE_DIM = 7
FINGER_JOINT_DIM = 12
DEX_ACTION_DIM = ARM_POSE_DIM + FINGER_JOINT_DIM


class DexOpenDrawerEnv(OpenDrawerEnv):
    """Open-drawer task with a dexterous hand instead of a parallel gripper."""

    def process_action(self, action: torch.Tensor | list[Dict]):
        if action is None:
            return None
        n = action.shape[0]
        cur = action.shape[1]
        if cur == DEX_ACTION_DIM:
            return action
        if cur < DEX_ACTION_DIM:
            pad = torch.zeros(
                (n, DEX_ACTION_DIM - cur), device=self.device, dtype=action.dtype
            )
            return torch.cat([action, pad], dim=1)
        return action[:, :DEX_ACTION_DIM]

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        robot_state = list(self.scene.robot_manager.get_robot_state()[0].values())[0]
        eef_pos = robot_state["eef_pos"]
        eef_quat = robot_state["eef_quat"]
        if eef_pos.dim() == 2:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        else:
            eef_pose = torch.cat([eef_pos, eef_quat], dim=-1)
        return eef_pose[env_ids]

    # ------------------------------------------------------------------ #
    # Dex trajectory loader (xhand_open_by_handle_trajectory format)
    # ------------------------------------------------------------------ #

    def get_drawer_trajectories(
        self,
        env_id: int,
        annotation_name: str = "xhand_open_by_handle_trajectory",
        joint_id: int = -1,
    ) -> dict:
        """Load 19-D world-frame trajectories from the xhand annotation.

        The annotation layout is::

            trajectories: { joint_<i>: { <traj_id>: { approach: [[19], …],
                                                       trajectory: [[19], …] } } }

        Each waypoint is ``[x, y, z, qw, qx, qy, qz, j0..j11]`` in the object's
        local frame. ``approach`` and ``trajectory`` are concatenated into a
        single ``(N, 19)`` tensor; the first 7 dims (pose) are transformed to
        world frame, finger joint targets stay as recorded.

        Returns ``{f"{joint}/{traj_id}": Tensor (N, 19)}`` (empty if missing).
        """
        obj = self.scene.scene_manager.articulation_objects[env_id][
            "articulation_items"
        ][0]
        joint_filter = None
        if joint_id is not None and joint_id >= 0:
            num_joints = obj.num_joints
            if joint_id >= num_joints:
                raise ValueError(
                    f"joint_id={joint_id} out of range: articulation has "
                    f"{num_joints} joints (valid: 0..{num_joints - 1})"
                )
            joint_filter = f"joint_{joint_id}"

        annotation_data = obj.get_annotation(annotation_name)
        if annotation_data is None:
            print(f"[DexOpenDrawerEnv] Warning: No '{annotation_name}' annotation")
            return {}

        trajs = annotation_data.get("trajectories")
        if not isinstance(trajs, dict):
            print("[DexOpenDrawerEnv] Warning: annotation has no 'trajectories' dict")
            return {}

        obj_pos, obj_quat, obj_rot_matrix, device = obj._get_object_transform()

        result: Dict[str, torch.Tensor] = {}
        for joint, joint_trajs in trajs.items():
            if joint_filter is not None and joint != joint_filter:
                continue
            if not isinstance(joint_trajs, dict):
                continue
            for traj_id, phases in joint_trajs.items():
                wps = self._extract_dex_waypoints(phases)
                if wps is None or wps.numel() == 0:
                    continue
                wps = wps.to(device=device, dtype=torch.float32)
                pose = obj._transform_poses_batch(
                    wps[:, :ARM_POSE_DIM], obj_pos, obj_quat, obj_rot_matrix
                )
                joints = wps[:, ARM_POSE_DIM : ARM_POSE_DIM + FINGER_JOINT_DIM]
                result[f"{joint}/{traj_id}"] = torch.cat([pose, joints], dim=1)
        return result

    @staticmethod
    def _extract_dex_waypoints(phases) -> torch.Tensor | None:
        """Concatenate ``approach`` + ``trajectory`` lists into one ``(N, 19)`` tensor."""
        rows: list[list[float]] = []
        if isinstance(phases, dict):
            for key in ("approach", "trajectory"):
                phase_data = phases.get(key)
                if isinstance(phase_data, list):
                    for wp in phase_data:
                        if isinstance(wp, list) and len(wp) == DEX_ACTION_DIM:
                            rows.append(wp)
        elif isinstance(phases, list):
            for wp in phases:
                if isinstance(wp, list) and len(wp) == DEX_ACTION_DIM:
                    rows.append(wp)
        if not rows:
            return None
        return torch.tensor(rows, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Termination — same range-based check as the parent OpenDrawerEnv.
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["scene_state"]["annotation"] = "xhand_open_by_handle_trajectory"
        return state
