import argparse

import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run custom DirectRL Env")
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Start SimulationApp
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import (
    FrankaCabinetEnv,
    FrankaCabinetEnvCfg,
)

from isaacsim.core.api.objects import DynamicCuboid
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import SingleDeformablePrim
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom
from isaacsim.core.utils.stage import get_current_stage


from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
from omni.physx.scripts import deformableUtils
from pxr import Gf


# reset color or texture
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np
from isaacsim.core.prims import SingleArticulation


def import_artic():
    add_reference_to_stage(
        "/home/magics/magicsim/MagicSim/Assets/Object/Articulated/ElectricKettle003.usd",
        "/Test/Articulation",
    )

    arti = SingleArticulation(prim_path="/Test/Articulation", position=(0, 0, 1))

    physics_sim_view = SimulationManager.get_physics_sim_view()
    arti.initialize(physics_sim_view)
    return None
    # return arti


def import_deform():
    mesh_path = "/World/Test/Deform_cube"
    skin_mesh = UsdGeom.Mesh.Define(get_current_stage(), mesh_path)
    tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(16)
    skin_mesh.GetPointsAttr().Set(tri_points)
    skin_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
    skin_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
    physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)
    physicsUtils.set_or_add_translate_op(skin_mesh, (0.0, 0.0, 2.0))
    deformable_material_path = "/World/Test/DeformableMaterial"
    deformable_material = DeformableMaterial(
        prim_path=deformable_material_path,
        dynamic_friction=0.5,
        youngs_modulus=5e4,
        poissons_ratio=0.4,
        damping_scale=0.1,
        elasticity_damping=0.1,
    )

    deformable = SingleDeformablePrim(
        name="deformablePrim",
        prim_path=str(mesh_path),
        deformable_material=deformable_material,
        vertex_velocity_damping=0.0,
        sleep_damping=1.0,
        sleep_threshold=0.05,
        settling_threshold=0.1,
        self_collision=True,
        self_collision_filter_distance=0.05,
        solver_position_iteration_count=20,
        kinematic_enabled=False,
        simulation_hexahedral_resolution=2,
        collision_simplification=True,
    )
    return deformable


def import_cloth():
    cloth_path = "/World/Test/Plane"

    plane_mesh = UsdGeom.Mesh.Define(get_current_stage(), cloth_path)
    tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(
        dimx=20, dimy=20, scale=1.0
    )
    plane_mesh.GetPointsAttr().Set(tri_points)
    plane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
    plane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
    init_loc = Gf.Vec3f(0, 0.0, 2.0)
    physicsUtils.setup_transform_as_scale_orient_translate(plane_mesh)
    physicsUtils.set_or_add_translate_op(plane_mesh, init_loc)

    particle_system_path = "/World/Test/particleSystem"
    particle_material_path = "/World/Test/particleMaterial"

    particle_material = ParticleMaterial(
        prim_path=str(particle_material_path), drag=0.1, lift=0.3, friction=0.6
    )
    radius = 0.5 * (0.6 / 5.0)
    restOffset = radius
    contactOffset = restOffset * 1.5
    particle_system = SingleParticleSystem(
        prim_path=str(particle_system_path),
        rest_offset=restOffset,
        contact_offset=contactOffset,
        solid_rest_offset=restOffset,
        fluid_rest_offset=restOffset,
        particle_contact_offset=contactOffset,
    )
    # note that no particle material is applied to the particle system at this point.
    # this can be done manually via self.particle_system.apply_particle_material(self.particle_material)
    # or to pass the material to the clothPrim which binds it internally to the particle system
    cloth = SingleClothPrim(
        name="clothPrim",
        prim_path=str(cloth_path),
        particle_system=particle_system,
        particle_material=particle_material,
    )
    return cloth


def import_cube():
    cube = DynamicCuboid(
        prim_path="/World/Test/Cube",
        scale=(0.2, 0.2, 0.2),
        position=(1, 0.0, 3.0),
    )
    return cube


def delete_deform():
    from isaacsim.core.utils.prims import delete_prim

    delete_prim("/World/Test/Deform_cube")
    delete_prim("/World/Test/DeformableMaterial")


def step_sim():
    for i in range(100):
        action = env.action_space.sample()
        print(action)
        obs, reward, terminated, truncated, info = env.step(torch.tensor(action))
        # print(f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")


def visual_cloth(cloth: SingleClothPrim):
    # from isaacsim.core.simulation_manager import SimulationManager

    # physics_sim_view = SimulationManager.get_physics_sim_view()
    # cloth_physics_physx_view: physx.ParticleClothView = (
    #     physics_sim_view.create_particle_cloth_view("/World/Plane")
    # )

    # reset color or texture
    from isaacsim.core.api.materials.preview_surface import PreviewSurface

    color = torch.tensor([0, 1, 0])
    visual_prim_path = "/World/Test/Plane/VisualMaterial"
    visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
    cloth.apply_visual_material(visual_material)

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    particle_pos = cloth._get_points_pose()
    # # stage = get_current_stage()
    # # geom_pts = UsdGeom.Points(stage.GetPrimAtPath("/World/Test/Plane"))
    # # particle_pos = geom_pts.GetPointsAttr().Get()  # for querying positions
    # # particle_pos = np.array(particle_pos)

    pcd.points = o3d.utility.Vector3dVector(particle_pos.cpu().numpy().reshape(-1, 3))
    o3d.visualization.draw_geometries([pcd])


def delete_cloth():
    from isaacsim.core.utils.prims import delete_prim

    delete_prim("/World/Test/Plane")
    delete_prim("/World/Test/particleSystem")
    delete_prim("/World/Test/particleMaterial")


def visual_cube(cube: DynamicCuboid):
    # reset color or texture
    from isaacsim.core.api.materials.preview_surface import PreviewSurface
    from isaacsim.core.simulation_manager import SimulationManager

    color = torch.tensor([1, 0, 0])
    visual_prim_path = "/World/Test/Cube/VisualMaterial"
    visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
    cube.apply_visual_material(visual_material)
    physics_sim_view = SimulationManager.get_physics_sim_view()
    cube._rigid_prim_view.initialize(physics_sim_view=physics_sim_view)

    cube._rigid_prim_view.set_world_poses(
        positions=torch.from_numpy(np.array([[2, 0.0, 3]]).astype(np.float32))
    )


def physics_cube(cube: DynamicCuboid):
    # reset physics material
    from isaacsim.core.api.materials.physics_material import PhysicsMaterial

    physics_prim_path = "/World/Test/Cube/PhysicsMaterial"
    physics_material = PhysicsMaterial(
        prim_path=physics_prim_path,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.1,
    )
    cube.apply_physics_material(physics_material)

    # reset size
    cube.set_size(1.5)

    # set physics contact offset
    cube.set_contact_offset(0.05)

    # set mass
    cube.set_mass(0.5)

    # set world position
    cube.set_world_pose(position=[2, 0.0, 3])

    step_sim()
    step_sim()

    for i in range(500):
        cube.set_linear_velocity(torch.tensor([0.0, 0.0, 1]))
        print(cube.get_world_pose())
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(torch.tensor(action))

    step_sim()


from isaacsim.core.api.objects import FixedCuboid


def import_geom():
    fix_cube = FixedCuboid(
        prim_path="/World/Test/FixedCube",
        scale=(0.2, 0.2, 0.2),
        position=(0.0, 0.0, 1.0),
    )
    return fix_cube


def visual_geom(fix_cube: FixedCuboid):
    from isaacsim.core.api.materials.preview_surface import PreviewSurface

    color = torch.tensor([0, 0, 1])
    visual_prim_path = "/World/Test/Cube/VisualMaterial_fix"
    visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
    fix_cube.apply_visual_material(visual_material)
    fix_cube.set_size(0.5)


def visual_deform(deform: SingleDeformablePrim):
    color = torch.tensor([0, 1, 1])
    visual_prim_path = "/World/Test/Deform_cube/VisualMaterial"
    visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
    deform.apply_visual_material(visual_material)

    # pcd = o3d.geometry.PointCloud()
    # particle_pos = deform._get_points_pose()

    # physics_sim_view = SimulationManager.get_physics_sim_view()
    # deform_physics_physx_view = physics_sim_view.create_soft_body_view(
    #     "/World/Test/Deform_cube"
    # )
    # deform._deformable_prim_view._physics_sim_view = physics_sim_view
    # deform._deformable_prim_view._physics_view = deform_physics_physx_view
    # particle_pos = deform._deformable_prim_view.get_simulation_mesh_nodal_positions()

    # for i in range(100):
    #     particle_pos = particle_pos.reshape(-1, 3)
    #     particle_pos[0, 2] += 0.05
    #     deform._deformable_prim_view.set_simulation_mesh_nodal_positions(particle_pos)
    #     for i in range(10):
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(torch.tensor(action))
    #     particle_pos = (
    #         deform._deformable_prim_view.get_simulation_mesh_nodal_positions()
    #     )


if __name__ == "__main__":
    cfg = FrankaCabinetEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cpu"
    cfg.sim.use_fabric = False
    cfg.scene.replicate_physics = False
    cfg.scene.clone_in_fabric = False
    env: FrankaCabinetEnv = gym.make("Isaac-Franka-Cabinet-Direct-v0", cfg=cfg)
    env.reset()
    for i in range(50):
        env.unwrapped.sim.app.update()
    while 1:
        env.unwrapped.sim.app.update()
    env.close()
