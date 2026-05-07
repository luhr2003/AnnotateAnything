from typing import Any, Dict, Sequence

import torch
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
import gymnasium as gym


class CloseDrawerEnv(TaskBaseEnv):
    """
    Close Drawer Environment for Robot Tasks.
    Uses close_by_push trajectory annotation.
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def get_policy_obs(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        camera_info = self.scene.capture_manager.step()
        return {"robot_state": robot_state, "camera_info": camera_info}

    def get_privilege_obs(self) -> Dict[str, Any]:
        articulation_pose = self._get_articulation_pose()
        return {"articulation_pose": articulation_pose}

    def _get_articulation_pose(
        self, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(
                self.scene.robot_manager.num_envs, device=self.device
            )
        poses = []
        for env_id in env_ids:
            translation, orientation = self.scene.scene_manager.articulation_objects[
                env_id
            ]["articulation_items"][0].get_local_pose()
            poses.append(torch.cat([translation, orientation], dim=0))
        return torch.stack(poses, dim=0)

    def process_action(self, action: torch.Tensor | list[Dict]):
        if action is None:
            return None
        if action.shape[1] < 8:
            action = torch.cat(
                [action, torch.zeros((action.shape[0], 1), device=self.device)], dim=1
            )
        else:
            action = action[:, :8]
        return action

    def get_eef_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        eef_pos = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_pos"
        ]
        eef_quat = list(self.scene.robot_manager.get_robot_state()[0].values())[0][
            "eef_quat"
        ]
        eef_pose = torch.cat([eef_pos, eef_quat], dim=1)
        return eef_pose[env_ids]

    def get_info(self) -> Dict[str, Any]:
        state = self.get_state()
        return {"state": state}

    def get_state(self) -> Dict[str, Any]:
        robot_state = self.scene.robot_manager.get_robot_state()
        articulation_pose = self._get_articulation_pose()
        state = {
            "robot_state": robot_state,
            "scene_state": {
                "articulation_pose": articulation_pose,
            },
            "camera_state": self.scene.camera_manager.get_all_camera_state(),
        }
        return state

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        reward = [0] * self.num_envs
        return reward

    # ------------------------------------------------------------------ #
    # Drawer data-access helpers (used by AtomicSkill/CloseDrawer)
    # ------------------------------------------------------------------ #

    def get_drawer_trajectories(
        self,
        env_id: int,
        annotation_name: str = "close_by_push_trajectory",
        joint_id: int = -1,
    ) -> dict:
        """Load world-frame trajectories from the articulation object's annotation.

        Args:
            env_id: Environment ID.
            annotation_name: Name of the annotation to load.
            joint_id: Joint index (0, 1, 2, ...). -1 means all joints.
                      Raises ValueError if joint_id >= num_joints.

        Returns:
            dict: {f"{joint}/{traj_id}": [N, 7] world-frame tensor}, empty dict
                  if no annotation or trajectory data is available.
        """
        obj = self.scene.scene_manager.articulation_objects[env_id][
            "articulation_items"
        ][0]
        joint_name = None
        if joint_id >= 0:
            num_joints = obj.num_joints
            if joint_id >= num_joints:
                raise ValueError(
                    f"joint_id={joint_id} out of range: articulation has {num_joints} joints (valid: 0..{num_joints - 1})"
                )
            joint_name = f"joint_{joint_id}"
        traj_data = obj.get_trajectory_poses(
            annotation_name=annotation_name,
            joint_name=joint_name,
            transform_to_world=True,
        )
        if traj_data is None:
            print(
                f"[CloseDrawerEnv] Warning: No '{annotation_name}' annotation on object"
            )
            return {}

        # Find the trajectory dict (supports both "trajectories" and
        # "grasp_trajectories" keys)
        trajs = None
        for key, value in traj_data.items():
            if isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, dict):
                        trajs = value
                        break
            if trajs is not None:
                break

        if trajs is None:
            print("[CloseDrawerEnv] Warning: No trajectory data found in annotation")
            return {}

        # Flatten into {joint/traj_id: tensor} format — already in world frame
        result = {}
        for joint, joint_trajs in trajs.items():
            if not isinstance(joint_trajs, dict):
                continue
            for traj_id, traj_tensor in joint_trajs.items():
                if isinstance(traj_tensor, torch.Tensor):
                    result[f"{joint}/{traj_id}"] = traj_tensor
        return result

    def get_drawer_object_pose(self, env_id: int):
        """Return (pos, quat, scale) for the drawer articulation object."""
        obj = self.scene.scene_manager.articulation_objects[env_id][
            "articulation_items"
        ][0]
        pos, quat = obj.get_local_pose()
        scale = torch.tensor(obj.init_scale, dtype=torch.float32)
        return pos, quat, scale

    def set_drawer_open(
        self, env_id: int, annotation_name: str = "close_by_push_trajectory"
    ):
        """Open the drawer joints to their initial angles from the annotation.

        Reads ``initial_joint_angles`` from the annotation and sets joint
        positions so the drawers start open.
        """
        obj = self.scene.scene_manager.articulation_objects[env_id][
            "articulation_items"
        ][0]
        traj_annotation = obj.get_annotation(annotation_name)
        if traj_annotation is None or "initial_joint_angles" not in traj_annotation:
            print(
                "[CloseDrawerEnv] Warning: no initial_joint_angles in annotation, "
                "skipping drawer open"
            )
            return

        init_angles = traj_annotation["initial_joint_angles"]
        joint_names = sorted(init_angles.keys())
        positions = [init_angles[jn] * 2 / 3 for jn in joint_names]
        obj.set_current_joint_positions(positions)
        print(
            f"[CloseDrawerEnv] Set drawer joints open: "
            f"{dict(zip(joint_names, positions))}"
        )

        # Let the simulation settle with the new positions
        for _ in range(10):
            self.scene.sim.sim_step()

    def reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset envs and open drawer for each reset env. Termination = drawer closed."""
        super().reset_idx(env_ids=env_ids, seed=seed, options=options)
        if env_ids is not None:
            env_id_list = (
                env_ids.detach().cpu().tolist()
                if isinstance(env_ids, torch.Tensor)
                else list(env_ids)
            )
            for eid in env_id_list:
                self.set_drawer_open(int(eid))

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate when any drawer joint has been pushed closed (progress <= 20%)."""
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_id in range(self.num_envs):
            obj = self.scene.scene_manager.articulation_objects[env_id][
                "articulation_items"
            ][0]
            current_pos = obj.get_current_joint_positions()
            lower = torch.as_tensor(obj.lower_joint_positions, dtype=torch.float32)
            upper = torch.as_tensor(obj.upper_joint_positions, dtype=torch.float32)
            joint_range = upper - lower
            valid = joint_range.abs() > 1e-6
            if valid.any():
                progress = (current_pos[valid] - lower[valid]) / joint_range[valid]
                # Closed = any joint within 5% of range
                if progress.min() <= 0.05:
                    termination[env_id] = True

        return termination, truncated
