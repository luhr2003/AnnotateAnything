import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncBaseEnv import SyncBaseEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="avatar_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    print(cfg)
    logger = Logger("TestAvatarEnv", log)
    env: SyncBaseEnv = gym.make(
        "SyncBaseEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    for j in range(5):
        print(f"Env Reset {j + 1}")
        for i in range(100):
            env.step()
        env.reset_idx()

    # Test Avatar command system
    print("\n=== Testing Avatar Command System with 4 Environments ===")

    # Check if animation_manager exists
    if hasattr(env, "animation_manager"):
        animation_manager = env.animation_manager

        # Inject initial commands to all avatars in all environments
        for env_id in range(animation_manager.num_envs):
            if (
                env_id in animation_manager.avatars
                and len(animation_manager.avatars[env_id]) > 0
            ):
                avatar = animation_manager.avatars[env_id][0]
                print(f"\nEnv {env_id}: {avatar.character_name}")

                # Different initial commands for each environment
                if env_id == 0:
                    avatar.inject_command(
                        [
                            ["GoTo", "5.0", "3.0", "0.0", "0.0"],
                            ["Sit", "3.0"],
                        ]
                    )
                    print("  - Injected: GoTo -> Sit")
                elif env_id == 1:
                    avatar.inject_command(
                        [
                            ["LookAround", "3.0"],
                            ["GoTo", "-3.0", "2.0", "0.0", "90.0"],
                        ]
                    )
                    print("  - Injected: LookAround -> GoTo")
                elif env_id == 2:
                    avatar.inject_command(
                        [
                            ["Talk", "3.0"],
                            ["Idle", "2.0"],
                        ]
                    )
                    print("  - Injected: Talk -> Idle")
                else:
                    avatar.inject_command(
                        [
                            ["Idle", "2.0"],
                            ["Sit", "3.0"],
                        ]
                    )
                    print("  - Injected: Idle -> Sit")

        # Execute initial commands
        print("\n=== Executing Initial Commands ===")
        for frame in range(200):
            env.step()
            if frame % 60 == 0:
                print(f"Frame {frame}/200")

        print("\n=== Initial Commands Complete ===")

    # Continuous loop with all 4 avatars
    print("\n=== Starting Continuous Loop for All 4 Avatars ===")
    print("Each avatar will independently execute random commands")
    print("Press Ctrl+C to exit\n")

    try:
        frame_count = 0
        # Track last inject time for each environment
        last_inject_time = {env_id: 0 for env_id in range(animation_manager.num_envs)}
        min_wait_frames = 30  # Minimum 0.5 seconds between checks

        while True:
            env.step()
            frame_count += 1

            # Check each environment independently
            if hasattr(env, "animation_manager"):
                for env_id in range(animation_manager.num_envs):
                    if env_id not in animation_manager.avatars:
                        continue

                    # Check if ready to inject new command
                    if frame_count - last_inject_time[env_id] >= min_wait_frames:
                        avatar = animation_manager.avatars[env_id][0]

                        # Check if command queue is empty
                        queue_len = avatar.get_command_queue_length()
                        current_action = avatar.get_current_action()

                        if queue_len == 0 and current_action == "None":
                            import random

                            # Randomly choose action
                            action_type = random.choice(
                                [
                                    "GoTo",
                                    "Sit",
                                    "Talk",
                                    "Idle",
                                    "LookAround",
                                    "PushButton",
                                ]
                            )

                            if action_type == "GoTo":
                                x = random.uniform(-5, 5)
                                y = random.uniform(-5, 5)
                                angle = random.uniform(0, 360)
                                avatar.inject_command(
                                    [["GoTo", str(x), str(y), "0.0", str(angle)]]
                                )
                                print(
                                    f"[Env{env_id}][Frame {frame_count}] Injected: GoTo({x:.1f}, {y:.1f}, {angle:.0f}°)"
                                )
                            else:
                                duration = random.uniform(2.0, 5.0)
                                avatar.inject_command([[action_type, str(duration)]])
                                print(
                                    f"[Env{env_id}][Frame {frame_count}] Injected: {action_type}({duration:.1f}s)"
                                )

                            last_inject_time[env_id] = frame_count

            # Print overall status every 120 frames (2 seconds)
            if frame_count % 120 == 0:
                print(f"\n[Frame {frame_count}] Status:")
                for env_id in range(animation_manager.num_envs):
                    if env_id in animation_manager.avatars:
                        avatar = animation_manager.avatars[env_id][0]
                        action = avatar.get_current_action()
                        queue = avatar.get_command_queue_length()
                        print(
                            f"  Env{env_id}: {avatar.character_name} - action={action}, queue={queue}"
                        )

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
