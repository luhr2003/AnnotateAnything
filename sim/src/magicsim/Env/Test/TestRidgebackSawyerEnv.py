import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


def print_robot_state(env: SyncRobotEnv, step_idx: int):
    """打印机器人状态，用于调试"""
    robot_states = env.robot_manager.get_robot_state()

    # robot_states 是一个列表，每个元素是一个字典 {robot_name: state_dict}
    for state_dict in robot_states:
        for robot_name, state in state_dict.items():
            print(f"\n=== Step {step_idx}: {robot_name} State ===")

            # 获取关节位置 (所有环境)
            joint_pos = state.get("joint_pos")
            if joint_pos is not None:
                # RidgebackFranka: base虚拟关节[0:3] + arm关节[3:10] + gripper[10:12]
                print(f"Joint positions shape: {joint_pos.shape}")
                print(f"  Base virtual joints [0:3]: {joint_pos[0, :3].cpu().numpy()}")
                print(f"  Arm joints [3:10]: {joint_pos[0, 3:10].cpu().numpy()}")
                print(f"  Gripper joints [10:12]: {joint_pos[0, 10:12].cpu().numpy()}")

            # 获取末端执行器位置
            eef_pos = state.get("eef_pos")
            eef_quat = state.get("eef_quat")
            if eef_pos is not None:
                print(f"  EEF position (env 0): {eef_pos[0].cpu().numpy()}")
            if eef_quat is not None:
                print(f"  EEF quaternion (env 0): {eef_quat[0].cpu().numpy()}")


def print_action_terms_state(env: SyncRobotEnv):
    """打印 ActionManager 中每个 action term 的状态"""
    print("\n=== ActionManager Terms State ===")
    for robot_name, am in env.robot_manager.action_managers.items():
        print(f"Robot: {robot_name}")
        for term_name, term in am._terms.items():
            print(f"  Term: {term_name}")
            print(f"    Type: {type(term).__name__}")
            print(f"    action_dim: {term.action_dim}")

            # env_ids 可能在 process_actions 调用前不存在
            env_ids = getattr(term, "_env_ids", None)
            print(f"    env_ids: {env_ids}")

            # 打印 raw_actions 和 processed_actions
            raw_actions = getattr(term, "_raw_actions", None)
            if raw_actions is not None:
                print(f"    raw_actions shape: {raw_actions.shape}")
                print(f"    raw_actions (env 0): {raw_actions[0].cpu().numpy()}")

            processed = getattr(term, "_processed_actions", None)
            if processed is not None:
                print(f"    processed_actions shape: {processed.shape}")
                if processed.numel() > 0:
                    print(
                        f"    processed_actions (env 0): {processed[0].cpu().numpy()}"
                    )

            # 打印 joint_ids
            joint_ids = getattr(term, "_joint_ids", None)
            joint_names = getattr(term, "_joint_names", None)
            if joint_ids is not None:
                print(f"    joint_ids: {joint_ids}")
            if joint_names is not None:
                print(f"    joint_names: {joint_names}")


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="ridgebackSawyerbase"
)
def main(cfg: DictConfig):
    print("cfg: ", cfg)
    # print(cfg)
    print(cfg.sim)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )

    env.reset()

    # 打印初始状态
    print_robot_state(env, step_idx=-1)
    print_action_terms_state(env)

    for i in range(50):
        rand_action_batched = env.robot_manager.sample_actions(batched=True)

        env.step(action=rand_action_batched)

    env.reset_idx([0, 3])

    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[1, 2]
        )
        env.step(action=rand_action_batched, env_ids=[1, 2])

    env.reset_idx()
    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(batched=True)
        env.step(action=rand_action_batched)

    env.reset_idx([0, 3])
    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 3]
        )
        env.step(action=rand_action_batched, env_ids=[0, 3])

    env.reset_idx([0, 3])
    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 2]
        )
        env.step(action=rand_action_batched, env_ids=[0, 2])

    while 1:
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 1]
        )
        env.step(action=rand_action_batched, env_ids=[0, 1])
        for i in range(10):
            env.sim.sim_step()

    env.close()


if __name__ == "__main__":
    main()
