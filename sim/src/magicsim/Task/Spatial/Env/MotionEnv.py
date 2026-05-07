import torch
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv
import gymnasium as gym


class MotionEnv(TaskCameraBaseEnv):
    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.config = config

    def reset(self):
        self.scene.reset()

    def sim_step(self):
        super().sim_step()

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def step(self):
        super().step()

    def get_policy_obs(self, env_ids):
        camera_info = self.scene.capture_manager.step()
        print(camera_info)
        return {"camera_info": camera_info}

    def get_movable_pose(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        movable_pose = []
        for env_id in env_ids:
            movable_pos, movable_ori = self.scene.scene_manager.rigid_objects[env_id][
                "Movable"
            ][0].get_local_pose()
            movable_pose.append(torch.cat([movable_pos, movable_ori], dim=0))

        if len(movable_pose) == 1:
            return movable_pose[0].unsqueeze(0)
        else:
            return torch.stack(movable_pose, dim=0)

    def get_privilege_obs(self, env_ids):
        movable_pose = self.get_movable_pose(env_ids)
        return {"movable_pose": movable_pose}

    def get_info(self):
        pass

    def get_termination(self):
        return torch.tensor([False] * self.num_envs, dtype=torch.bool), torch.tensor(
            [False] * self.num_envs, dtype=torch.bool
        )

    def get_reward(self, action, env_ids):
        return torch.tensor([0] * self.num_envs, dtype=torch.float32)
