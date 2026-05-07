import torch
from magicsim.Task.TableTop.Env.CloseDrawerEnv import CloseDrawerEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from loguru import logger as log

# Annotation: close_by_push trajectory.json (key: "close_by_push trajectory")
CLOSE_BY_PUSH_ANNOTATION = "close_by_push_trajectory"


@hydra.main(version_base=None, config_path="../../Conf", config_name="close_drawer_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: CloseDrawerEnv = gym.make(
        "CloseDrawerEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    # Let the simulation settle
    for i in range(40):
        env.sim_step()

    # Get the articulation object (env 0, first articulation item)
    articulation_obj = env.scene.scene_manager.articulation_objects[0][
        "articulation_items"
    ][0]

    # Set initial joint positions to init_angles / 2 (half of annotation initial)
    traj_annotation = articulation_obj.get_annotation(CLOSE_BY_PUSH_ANNOTATION)
    init_angles = (
        traj_annotation["initial_joint_angles"]
        if traj_annotation and "initial_joint_angles" in traj_annotation
        else None
    )
    upper_joint_pos = articulation_obj.upper_joint_positions
    if isinstance(upper_joint_pos, torch.Tensor):
        upper_joint_pos = upper_joint_pos.cpu().tolist()
    else:
        upper_joint_pos = list(upper_joint_pos)
    if init_angles is not None:
        positions = [
            init_angles["joint_0"] / 2,
            init_angles["joint_1"] / 2,
            init_angles["joint_2"] / 2,
        ]
        articulation_obj.set_current_joint_positions(positions)
    print(f"upper_joint_pos: {upper_joint_pos}")
    print(f"init_angles from annotation: {init_angles}")

    # Get trajectory poses (transformed to world coordinates) for all joints
    traj_data = articulation_obj.get_trajectory_poses(
        annotation_name=CLOSE_BY_PUSH_ANNOTATION,
        joint_name=None,  # get all joints
        transform_to_world=True,
    )
    print(f"Trajectory data keys: {list(traj_data.keys())}")

    # Support both "trajectories" and "grasp_trajectories" keys
    trajs = traj_data.get("trajectories") or traj_data.get("grasp_trajectories")

    # Show trajectories for joint_0, joint_1, joint_2
    for joint_idx, joint_name in enumerate(["joint_0", "joint_1", "joint_2"]):
        if joint_name not in trajs:
            print(f"Warning: {joint_name} not in trajectories, skipping")
            continue
        joint_trajs = trajs[joint_name]
        first_traj_key = sorted(joint_trajs.keys())[0]
        first_traj = joint_trajs[first_traj_key]  # Tensor (N, 7)
        first_traj = torch.tensor(first_traj)
        print(
            f"{joint_name} trajectory key: {first_traj_key}, shape: {first_traj.shape}"
        )
        print(f"{joint_name} trajectory waypoints: ", first_traj)

        # Visualize the trajectory waypoints as axes (clear only for first joint)
        draw_grasp_samples_as_axes(
            grasp_poses=first_traj,
            axis_length=0.03,
            line_thickness=3,
            line_opacity=0.8,
            clear_existing=(joint_idx == 0),
        )

    # Keep simulation running to observe visualization
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
