# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the contact sensor.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import argparse

import numpy as np
import torch
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import RigidPrim, SingleGeometryPrim
import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import SimulationContext
from isaacsim.core.utils.stage import add_reference_to_stage
import carb


class RigidViewExample:
    def __init__(self):
        self._array_container = torch.Tensor
        self.sim_cfg = SimulationCfg(device="cpu", use_fabric=False)
        # self.sim = SimulationContext(stage_units_in_meters=1.0, backend="torch")
        self.sim = SimulationContext(self.sim_cfg)
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/disableContactProcessing", False)
        self.stage = simulation_app.context.get_stage()
        # Ground-plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        # Lights
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        cfg.func("/World/Light", cfg)

    def makeEnv(self):
        self.sim.reset(soft=False)
        for i in range(10):
            self.sim.step()
        self.cube_height = 1.0
        self.top_cube_height = self.cube_height + 3.0
        self.cube_dx = 5.0
        self.cube_y = 2.0
        self.top_cube_y = self.cube_y + 0.0

        FixedCuboid(
            prim_path="/World/Room/See/Box_1",
            name=f"box_{i}",
            size=1.0,
            color=np.array([0.5, 0, 0]),
            position=[0, 0, 0.5],
        )
        add_reference_to_stage(
            usd_path="/home/ubuntu/magicsim/MagicSim/Assets/Sensor/Camera/kinect.usd",
            prim_path="/World/Camera",
        )

        room = SingleGeometryPrim(prim_path="/World/Room", collision=True)

        room.initialize(self.sim.physics_sim_view)

        self.camera = SingleGeometryPrim(
            prim_path="/World/Camera", name="camera", collision=True
        )
        from isaacsim.core.simulation_manager import SimulationManager

        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        # a view just to manipulate the top boxes
        self._top_box_view = RigidPrim(
            prim_paths_expr="/World/Camera",
            name="top_box_view",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=["/World/Room/See/*"],
        )
        self._top_box_view.initialize(self.physics_sim_view)
        self._top_box_view.disable_gravities()

        self.camera.initialize(self.physics_sim_view)

    def play(self):
        self.makeEnv()
        reset_needed = False
        while simulation_app.is_running():
            self.camera.set_world_pose([0, 0, 1.5])
            for i in range(30):
                self.sim.step(render=True)

            cur_pose, _ = self.camera.get_world_pose()
            self.camera.set_world_pose([0, 0, 0.5])

            flag = False

            print("New iteration")
            for i in range(5):
                # net_forces = self._box_view.get_net_contact_forces(None, dt=1 / 60)
                forces_matrix = self._top_box_view.get_contact_force_matrix(
                    None, dt=1 / 60
                )
                top_net_forces = self._top_box_view.get_net_contact_forces(
                    None, dt=1 / 60
                )
                # print("Bottom box net forces: \n", net_forces)
                # print("Top box net forces: \n", top_net_forces)
                # print("Bottom box forces from top ones: \n", forces_matrix)

                if top_net_forces.any() > 0 or forces_matrix.any() > 0:
                    flag = True
                    print(self.sim.current_time_step_index)
                    print("Top box is in contact with the bottom box")
                    print("Top box net forces: \n", top_net_forces)
                    print("Top box forces from bottom ones: \n", forces_matrix)
                self.sim.step()

            for i in range(100):
                self.sim.step(render=True)
        simulation_app.close()


RigidViewExample().play()
