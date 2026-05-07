import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import hydra
import cv2
import torch
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF

import threading
import numpy as np

# --- 全局变量 (保持不变) ---
g_left_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
g_right_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
g_image_lock = threading.Lock()
g_stop_event = threading.Event()
# ---------------------------------


# --- 3. [已修改] 创建 CV2 显示函数 ---
def cv2_display_thread(target_height=480):
    """
    此函数在单独的线程中运行。
    它处理所有的 cv2.imshow 和 cv2.waitKey 调用，
    并自动将小图像放大以便查看。
    """
    global g_left_rgb, g_right_rgb, g_image_lock, g_stop_event

    window_name = "Dual-Column Real-time Feed (Press 'q' to quit)"
    # WINDOW_NORMAL 允许用户在放大后再次手动调整窗口大小
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while not g_stop_event.is_set():
        with g_image_lock:
            local_left = g_left_rgb.copy()
            local_right = g_right_rgb.copy()

        try:
            # 确保图像有效
            if local_left.shape[0] > 0 and local_right.shape[0] > 0:
                # --- [核心修改] ---
                # 1. 放大左侧图像
                h_l, w_l, _ = local_left.shape
                # 如果高度太小，才进行缩放
                if h_l < target_height:
                    scale_l = target_height / h_l
                    new_w_l = int(w_l * scale_l)
                    # 使用 INTER_NEAREST 保持像素感，适合传感器数据
                    local_left_resized = cv2.resize(
                        local_left,
                        (new_w_l, target_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    local_left_resized = local_left

                # 2. 放大右侧图像
                h_r, w_r, _ = local_right.shape
                if h_r < target_height:
                    scale_r = target_height / h_r
                    new_w_r = int(w_r * scale_r)
                    local_right_resized = cv2.resize(
                        local_right,
                        (new_w_r, target_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    local_right_resized = local_right
                # --- [修改结束] ---

                # 3. 合并放大后的图像
                combined_image = np.hstack((local_left_resized, local_right_resized))

                cv2.imshow(window_name, combined_image)

        except Exception as e:
            log.warning(f"CV2 display error: {e}")
            pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("CV2 thread: 'q' pressed, signaling stop.")
            g_stop_event.set()
            break

    cv2.destroyAllWindows()
    log.info("CV2 thread: Stopped.")


# ----------------------------------------------------


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="tactile_config")
def main(cfg: DictConfig):
    global g_left_rgb, g_right_rgb, g_image_lock, g_stop_event

    new_seed = int(13)
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)
    print(cfg)
    if "robot" in cfg and cfg.robot is not None:
        try:
            tactile_cfg_yaml = OmegaConf.to_yaml(cfg.robot.get("Tactile"))
            log.info("Loaded tactile configuration:\n{}", tactile_cfg_yaml)
        except Exception as e:
            log.warning(f"Failed to print tactile configuration: {e}")
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # --- 5. [轻微修改] 启动 CV2 显示线程 ---
    # 在这里，你可以通过 args=(N,) 来设置你想要的高度
    # 默认是 480 像素高
    display_thread = threading.Thread(
        target=cv2_display_thread, args=(480,), daemon=True
    )
    display_thread.start()
    # ----------------------------------

    num = 1
    first_time = True
    try:
        # --- 6. 运行主仿真循环 (保持不变) ---
        while not g_stop_event.is_set():
            if first_time:
                first_time = False
                for i in range(200):
                    object_handle = env.scene_manager.get_objects(
                        env_ids=[0, 1, 2, 3],
                        object_type="rigid",
                        object_name="Sugar_Box",
                    )
                    object_handle = list(object_handle.values())
                    object_handles = [h[0] for h in object_handle]
                    object_positions = [h.get_local_pose()[0] for h in object_handles]
                    print(f"Object positions: {object_positions}")
                    grasp_pose_goal_list = []
                    for pos in object_positions:
                        grasp_pose_goal_list.append(
                            torch.tensor(
                                [pos[0], pos[1], pos[2] - 1, 0, 0, 1, 0, 0],
                                device=env.device,
                            )
                        )

                    grasp_pose_goal = torch.stack(grasp_pose_goal_list)
                    print(f"Grasp pose goal: {grasp_pose_goal}")
                    env.step(action=grasp_pose_goal)

                    left_rgb_tensor = (
                        env.robot_manager.tactile_manager.get_sensor_data()[0]
                        .output["tactile_rgb"][0]
                        .cpu()
                        .numpy()
                    )
                    right_rgb_tensor = (
                        env.robot_manager.tactile_manager.get_sensor_data()[1]
                        .output["tactile_rgb"][0]
                        .cpu()
                        .numpy()
                    )

                    # --- 7. 更新全局图像 (保持不变) ---
                    with g_image_lock:
                        g_left_rgb = left_rgb_tensor
                        g_right_rgb = right_rgb_tensor

                    # --------------------------------------

            for i in range(100):
                object_handle = env.scene_manager.get_objects(
                    env_ids=[0, 1, 2, 3], object_type="rigid", object_name="Sugar_Box"
                )
                object_handle = list(object_handle.values())
                object_handles = [h[0] for h in object_handle]
                object_positions = [h.get_local_pose()[0] for h in object_handles]
                print("Close")
                grasp_pose_goal_list = []
                for pos in object_positions:
                    grasp_pose_goal_list.append(
                        torch.tensor(
                            [pos[0], pos[1], pos[2] - 1, 0, 0, 1, 0, 1],
                            device=env.device,
                        )
                    )

                grasp_pose_goal = torch.stack(grasp_pose_goal_list)
                print(f"Grasp pose goal: {grasp_pose_goal}")
                env.step(action=grasp_pose_goal)

                left_rgb_tensor = (
                    env.robot_manager.tactile_manager.get_sensor_data()[0]
                    .output["tactile_rgb"][0]
                    .cpu()
                    .numpy()
                )
                right_rgb_tensor = (
                    env.robot_manager.tactile_manager.get_sensor_data()[1]
                    .output["tactile_rgb"][0]
                    .cpu()
                    .numpy()
                )

                # --- 7. 更新全局图像 (保持不变) ---
                with g_image_lock:
                    g_left_rgb = left_rgb_tensor
                    g_right_rgb = right_rgb_tensor
                # --------------------------------------

            for i in range(100):
                object_handle = env.scene_manager.get_objects(
                    env_ids=[0, 1, 2, 3], object_type="rigid", object_name="Sugar_Box"
                )
                object_handle = list(object_handle.values())
                object_handles = [h[0] for h in object_handle]
                object_positions = [h.get_local_pose()[0] for h in object_handles]
                print(f"Object positions: {object_positions}")
                grasp_pose_goal_list = []
                for pos in object_positions:
                    grasp_pose_goal_list.append(
                        torch.tensor(
                            [pos[0], pos[1], pos[2] - 1, 0, 0, 1, 0, 0],
                            device=env.device,
                        )
                    )

                grasp_pose_goal = torch.stack(grasp_pose_goal_list)
                print("Open")
                env.step(action=grasp_pose_goal)

                left_rgb_tensor = (
                    env.robot_manager.tactile_manager.get_sensor_data()[0]
                    .output["tactile_rgb"][0]
                    .cpu()
                    .numpy()
                )
                right_rgb_tensor = (
                    env.robot_manager.tactile_manager.get_sensor_data()[1]
                    .output["tactile_rgb"][0]
                    .cpu()
                    .numpy()
                )

                # --- 7. 更新全局图像 (保持不变) ---
                with g_image_lock:
                    g_left_rgb = left_rgb_tensor
                    g_right_rgb = right_rgb_tensor
                # --------------------------------------

    except KeyboardInterrupt:
        log.info("Main thread: KeyboardInterrupt received. Signaling stop.")
        g_stop_event.set()
    except Exception as e:
        log.error(f"Main thread: An error occurred: {e}")
        g_stop_event.set()
    finally:
        # --- 8. 清理 (保持不变) ---
        log.info("Main thread: Stop signal received. Cleaning up.")
        display_thread.join()
        env.close()
        log.info("Main thread: Exiting.")
        # ---------------------------


if __name__ == "__main__":
    main()
