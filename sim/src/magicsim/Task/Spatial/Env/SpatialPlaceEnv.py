import torch
from magicsim.StardardEnv.Camera.TaskCameraBaseEnv import TaskCameraBaseEnv
import gymnasium as gym
from magicsim.Env.Scene.Object.Geometry import GeometryObject
from magicsim.Env.Animation.Avatar import Avatar


class SpatialPlaceEnv(TaskCameraBaseEnv):
    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)
        self.config = config

    def reset(self):
        self.scene.reset()
        self.house = self.scene.nav_manager.rooms[0]
        self.room_ids = self.house.list_room_ids()
        print(f"Room ids: {self.room_ids}")

    def sim_step(self):
        super().sim_step()

    def get_obs_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    def step(self):
        self.scene.camera_manager.set_camera_pose(
            "camera0", torch.tensor([[0, 0, 1, 0, 0, 0, 0]]), env_ids=[0]
        )
        for room_id in self.room_ids:
            room_annotation = self.house.get_room_by_id(room_id)

            room_free_points = self.house.get_room_free_point(
                room_id, height=1, radius=1
            )
            print(f"Room {room_id} has {len(room_free_points)} free points")
            room_free_points_world = torch.from_numpy(
                room_free_points + room_annotation["center_in_world"][:2]
            )

            room_free_points_world = torch.cat(
                [
                    room_free_points_world,
                    0.3 * torch.ones(room_free_points_world.shape[0], 1),
                ],
                dim=1,
            )

            robot: GeometryObject = self.scene.scene_manager.geometry_objects[0][
                "robot"
            ][0]
            robot.set_local_pose(room_free_points_world[3], torch.tensor([1, 0, 0, 0]))

            avatar: Avatar = self.scene.animation_manager.avatars[0][0]
            avatar.init_character()
            avatar.set_world_pose(room_free_points_world[9], torch.tensor([1, 0, 0, 0]))

            for i in range(200):
                self.sim_step()
            self.get_obs()

    def get_obs(self):
        obs = self.scene.capture_manager.step()
        # print(obs)
