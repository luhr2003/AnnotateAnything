from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Sensor.OccupancyManager import OccupancyManager
from magicsim.Env.Planner.Planner import Planner

import torch
import torch.nn.functional as F
import math
import numpy as np

import matplotlib.pyplot as plt

from magicsim.Env.Planner.Utils import angle_diff


def _get_cfg_val(obj, name: str, default):
    """从 OmegaConf / dict 取属性，兼容 getattr 和 []。"""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


class Dwb(Planner):
    def __init__(
        self,
        max_speed: float = 2.0,  # 最大前进线速度 [m/s]
        min_speed: float = -2.0,  # 最小线速度（可以允许倒车的话设为负数）
        max_steer: float = 0.8,  # 最大转向角 [rad]
        min_steer: float = -0.8,  # 最小转向角 [rad]
        max_accel_vx: float = 2.0,  # 最大线加速度 [m/s^2]
        min_vx: float = -2.0,  # 最小线速度 [m/s]
        max_vx: float = 2.0,  # 最大横向速度 [m/s]
        max_accel_vy: float = 2.0,  # 最大横向加速度 [m/s^2]
        min_vy: float = -2.0,  # 最小横向速度 [m/s]
        max_vy: float = 2.0,  # 最大横向速度 [m/s]
        max_yaw_accel: float = 2.0,  # 最大转向角加速度 [rad/s^2]
        min_yaw_rate: float = -2.0,  # 最小转向角速度 [rad/s]
        max_yaw_rate: float = 2.0,  # 最大转向角速度 [rad/s]
        num_vx_samples: int = 50,  # 线速度采样点数
        num_vy_samples: int = 50,  # 横向速度采样点数
        num_yaw_samples: int = 50,  # 转向角采样点数
        num_speed_samples: int = 50,  # 线速度采样点数
        num_steer_samples: int = 50,  # 转向角采样点数
        max_accel: float = 2.0,  # 线加速度约束 [m/s^2]
        max_steer_rate: float = 2.0,  # 转向角变化率约束 [rad/s]
        dt: float = 0.0167,  # 单次控制周期 [s]
        robot_manager: RobotManager = None,
        robot_name: str = "mobile",
        device: torch.device = torch.device("cpu"),
        occupancy_manager: OccupancyManager = None,
        wheel_base: float = 1.65,
        robot_radius: float = 0.3,
        controller_type: str = "unicycle",
        num_rollout_steps: int = 60,
        w_goal: float = 3.0,
        w_obs: float = 10.0,
        w_vel: float = 4.0,
        w_path: float = 3.0,
        planner_cfg=None,
        robot_config=None,
        base_planner_type: str = None,
        base_planner_config=None,
    ):
        self.robot_manager = robot_manager
        self.occupancy_manager = occupancy_manager
        self.robot_name = robot_name

        # 若传入 planner 配置，则从配置中解析 dwb_holonomic / dwb_humanoid 参数并覆盖默认值
        if (
            planner_cfg is not None
            and robot_config is not None
            and base_planner_type is not None
            and base_planner_config is not None
        ):
            key = base_planner_type.lower()
            if key not in (
                "dwb_holonomic",
                "dwb_humanoid",
                "dwb_quadruped",
                "dwb_differential",
            ):
                raise ValueError(
                    f"Dwb from config only supports dwb_holonomic or dwb_humanoid, got {key}"
                )
            low, high = (
                planner_cfg.base_action_space[key][0],
                planner_cfg.base_action_space[key][1],
            )
            min_vx = low[0].item()
            max_vx = high[0].item()
            min_vy = low[1].item()
            max_vy = high[1].item()
            min_speed = low[0].item()
            max_speed = high[0].item()
            min_steer = low[1].item()
            max_steer = high[1].item()

            dp = (
                base_planner_config.get(key, None)
                or getattr(base_planner_config, key, None)
                or {}
            )
            max_accel_vx = _get_cfg_val(dp, "max_accel_vx", 2.0)
            max_accel_vy = _get_cfg_val(dp, "max_accel_vy", 2.0)
            max_accel = _get_cfg_val(dp, "max_accel", 2.0)
            max_steer_rate = _get_cfg_val(dp, "max_steer_rate", 2.0)
            max_yaw_accel = _get_cfg_val(dp, "max_yaw_accel", 2.0)
            # Robot physics: prefer robot_config.physics (e.g. dwb_g1.yaml), else
            # fall back to the dwb sub-config so configs that don't ship a physics
            # block (e.g. single_g1.yaml used by SquatGrasp + Nav) still work.
            physics_cfg = _get_cfg_val(robot_config, "physics", None)
            wheel_base = _get_cfg_val(
                physics_cfg, "wheel_base", _get_cfg_val(dp, "wheel_base", 0.5)
            )
            robot_radius = _get_cfg_val(
                physics_cfg, "robot_radius", _get_cfg_val(dp, "robot_radius", 0.46)
            )
            if key == "dwb_differential":
                controller_type = "unicycle"
                min_yaw_rate = None
                max_yaw_rate = None
            elif (
                key == "dwb_holonomic"
                or key == "dwb_humanoid"
                or key == "dwb_quadruped"
            ):
                controller_type = "holonomic"
                min_yaw_rate = low[2].item()
                max_yaw_rate = high[2].item()
            else:
                raise ValueError(f"Unsupported controller type: {key}")
            num_rollout_steps = _get_cfg_val(dp, "num_rollout_steps", 60)
            w_goal = _get_cfg_val(dp, "w_goal", 3.0)
            w_obs = _get_cfg_val(dp, "w_obs", 10.0)
            w_vel = _get_cfg_val(dp, "w_vel", 4.0)
            w_path = _get_cfg_val(dp, "w_path", 3.0)
            w_vy = _get_cfg_val(dp, "w_vy", 0.0)
            turn_yaw_threshold = _get_cfg_val(dp, "turn_yaw_threshold", 999.0)
            dt = _get_cfg_val(dp, "dt", 0.0167)
            dilate_radius = _get_cfg_val(dp, "dilate_radius", 0.4)
            num_vx_samples = _get_cfg_val(dp, "num_vx_samples", 50)
            num_vy_samples = _get_cfg_val(dp, "num_vy_samples", 50)
            num_yaw_samples = _get_cfg_val(dp, "num_yaw_samples", 50)

            num_speed_samples = _get_cfg_val(dp, "num_speed_samples", 50)
            num_steer_samples = _get_cfg_val(dp, "num_steer_samples", 50)
            robot_type = _get_cfg_val(dp, "robot_type", "fucking nothing")
            # Debug / occupancy regen frequency / device — all overridable.
            # Defaults: debug=True (verbose during dev), occupancy regen every 200
            # Dwb.step calls (was 20 — too aggressive for collect's slower per-step).
            debug = bool(_get_cfg_val(dp, "debug", True))
            occupancy_update_interval = int(
                _get_cfg_val(dp, "occupancy_update_interval", 200)
            )
            device_str = _get_cfg_val(dp, "device", None)
            if device_str is not None:
                device = torch.device(str(device_str))

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steer = max_steer
        self.min_steer = min_steer

        self.max_accel_vx = max_accel_vx
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.max_accel_vy = max_accel_vy
        self.min_vy = min_vy
        self.max_vy = max_vy
        self.max_yaw_accel = max_yaw_accel
        self.min_yaw_rate = min_yaw_rate
        self.max_yaw_rate = max_yaw_rate

        self.num_vx_samples = num_vx_samples
        self.num_vy_samples = num_vy_samples
        self.num_yaw_samples = num_yaw_samples
        self.num_speed_samples = num_speed_samples
        self.num_steer_samples = num_steer_samples

        self.max_accel = max_accel
        self.max_steer_rate = max_steer_rate
        self.dilate_radius = dilate_radius
        # print("dilate_radius: ", dilate_radius)
        # raise Exception("Stop here")
        self.wheel_base = wheel_base
        self.dt = dt
        self.robot_radius = robot_radius
        self.w_goal = w_goal  # 目标代价权重
        self.w_obs = w_obs  # 碰撞代价权重
        self.w_vel = w_vel  # 速度代价权重（鼓励向前）
        self.w_path = w_path
        self.w_vy = w_vy  # 横向速度惩罚权重（抑制斜走）
        self.turn_yaw_threshold = (
            turn_yaw_threshold  # yaw 偏差超过此值进入 turning-only 模式
        )

        self.device = torch.device(device)
        self.num_rollout_steps = num_rollout_steps
        # ====== 地图参数（和 generate_occupancy 保持一致）======
        self.room_size = 16.0
        self.map_resolution = 0.05  # 每格 5cm

        self.controller_type = controller_type

        self.step_counter = 0
        self.robot_type = robot_type
        # Debug + occupancy regen interval (config-driven via dp dict above).
        # Defaults: debug=True (verbose during dev), interval=200 (regen rarely).
        self.debug: bool = bool(debug) if "debug" in dir() else True
        self.occupancy_update_interval: int = (
            int(occupancy_update_interval)
            if "occupancy_update_interval" in dir()
            else 200
        )
        # If we own this OccupancyManager and debug is on, surface its prints too.
        if (
            self.debug
            and self.occupancy_manager is not None
            and hasattr(self.occupancy_manager, "verbose")
        ):
            self.occupancy_manager.verbose = True

    @torch.no_grad()
    def generate_occupancy(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        为指定的环境生成占用地图，返回批量torch tensor。

        Args:
            env_ids: 环境ID tensor，形状为 [B]

        Returns:
            occupancy: torch.Tensor，形状为 [B, H, W]
                    0=占用，1=空闲，2=未知
        """
        device = self.device

        # ---------- env_ids 处理 ----------
        env_ids = env_ids.to(device)

        # 转换为 Python 列表，用于 OccupancyManager API
        env_ids_list = env_ids.cpu().tolist()
        num_envs_to_generate = len(env_ids_list)

        # ---------- 以机器人为中心的 rolling window boundary ----------
        half_size = self.room_size / 2.0

        # 机器人当前状态（world/env-local 坐标）
        robot_state = self.robot_manager.get_robot_state()[0][self.robot_name]
        base_pos_all = robot_state["base_pos"].to(device)  # [N_env, 3]
        base_pos = base_pos_all[env_ids]  # [B, 3]
        # print("base_pos: ", base_pos)
        boundaries = []
        scan_origins = []
        for i in range(num_envs_to_generate):
            x = base_pos[i, 0].item()
            y = base_pos[i, 1].item()

            # 这个 boundary 是在 env-local/world 坐标下，以车为中心的窗口
            boundary = [
                x - half_size,  # min_x
                x + half_size,  # max_x
                y - half_size,  # min_y
                y + half_size,  # max_y
                0.2,  # min_z
                0.5,  # max_z
            ]
            boundaries.append(boundary)

            # 扫描原点放在车上方
            scan_origin = [x, y, 0]
            scan_origins.append(scan_origin)

        if self.debug:
            print(
                f"Generating occupancy maps for {num_envs_to_generate} environments..."
            )

        # ---------- 生成占用图 ----------
        grids = self.occupancy_manager.generate(
            origin=scan_origins, boundary=boundaries, type="2d", env_ids=env_ids_list
        )

        # 处理返回值：可能是单个数组或列表
        if isinstance(grids, np.ndarray):
            # 单个环境的情况
            grid_list = [grids]
        else:
            # 多个环境的情况
            grid_list = grids

        # 检查所有网格是否具有相同的形状
        shapes = [g.shape for g in grid_list]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"Occupancy grids have inconsistent shapes: {shapes}. "
                f"All environments must have the same grid dimensions."
            )

        # ---------- 转成 torch batch ----------
        grid_tensors = [
            torch.from_numpy(grid).to(device=device, dtype=torch.float32)
            for grid in grid_list
        ]

        occupancy_batch = torch.stack(grid_tensors, dim=0)  # [B, H, W]

        # Allocate against total num_envs so subset env_ids (e.g. when Nav dispatches
        # only a subset to Dwb) index safely.
        num_envs_total = base_pos_all.shape[0]
        if (
            not hasattr(self, "map_meta_local_all")
            or self.map_meta_local_all is None
            or self.map_meta_local_all.shape[0] != num_envs_total
        ):
            self.map_meta_local_all = torch.zeros(num_envs_total, 3, device=device)

        # 对当前这批 env，写入各自的 origin_x, origin_y
        # 对应 boundary: [min_x, max_x, min_y, max_y, min_z, max_z]
        origin_x_list = [b[0] for b in boundaries]
        origin_y_list = [b[2] for b in boundaries]
        origin_x = torch.tensor(origin_x_list, device=device, dtype=torch.float32)
        origin_y = torch.tensor(origin_y_list, device=device, dtype=torch.float32)

        # 更新 map_meta_local_all[env_ids] = [origin_x, origin_y, resolution]
        self.map_meta_local_all[env_ids, 0] = origin_x
        self.map_meta_local_all[env_ids, 1] = origin_y
        self.map_meta_local_all[env_ids, 2] = self.map_resolution

        # vis env 0 occupancy map
        # vis_grid = (1 - occupancy_batch[0]) * 255
        # vis_grid = vis_grid.cpu().numpy().astype(np.uint8)
        # plt.imshow(vis_grid, cmap="gray")
        # plt.show()
        # ---------- 清掉机器人自身 footprint ----------
        occupancy_batch = self.clear_robot_footprint(
            occupancy_batch,
            robot_state=robot_state,
            env_ids=env_ids,
            radius=self.robot_radius,  # 根据你车的实际尺寸调
        )
        # vis env 0 occupancy map
        # vis_grid = (1 - occupancy_batch[0]) * 255
        # vis_grid = vis_grid.cpu().numpy().astype(np.uint8)
        # plt.imshow(vis_grid, cmap="gray")
        # plt.show()
        return occupancy_batch  # [B, H, W]

    @torch.no_grad()
    def clear_robot_footprint(self, occupancy, robot_state, env_ids, radius=0.6):
        """
        并行清空机器人自身 footprint 区域，防止车被当成障碍物。

        occupancy: [B, H, W]  0=free, 1=occupied
        robot_state: dict, 至少包含 "base_pos": [N_env, 3]
        env_ids: [B]，本次 batch 的 env id
        radius: 机器人半径（米）
        """
        device = occupancy.device
        B, H, W = occupancy.shape

        # 1) 取出这批 env 的机器人世界坐标
        base_pos = robot_state["base_pos"].to(device)  # [N_env, 3]
        base_pos = base_pos[env_ids]  # [B, 3]
        robot_x = base_pos[:, 0].view(B, 1, 1)  # [B,1,1]
        robot_y = base_pos[:, 1].view(B, 1, 1)  # [B,1,1]

        # 2) 每个 env 自己的 map_meta_local_all: [num_envs, 3] -> 这里取当前 batch 的 [B, 3]
        #   [origin_x, origin_y, resolution]
        map_meta = self.map_meta_local_all[env_ids].to(device)  # [B, 3]
        origin_x = map_meta[:, 0].view(B, 1, 1)  # [B,1,1]
        origin_y = map_meta[:, 1].view(B, 1, 1)  # [B,1,1]
        resolution = map_meta[:, 2].view(B, 1, 1)  # [B,1,1]
        # 如果所有 env 分辨率一样，也可以直接用 self.map_resolution，逻辑一样

        # 3) 构造像素坐标索引（不带 origin），再加上每个 env 的 origin
        idx_x = torch.arange(W, device=device).view(1, 1, W)  # [1,1,W]
        idx_y = torch.arange(H, device=device).view(1, H, 1)  # [1,H,1]

        # 世界坐标下的每个 cell 中心位置 (广播到 B)
        grid_x = origin_x + idx_x * resolution  # [B,H,W]
        grid_y = origin_y + idx_y * resolution  # [B,H,W]

        # 4) 计算每个像素到机器人位置的距离
        dx = grid_x - robot_x  # [B,H,W]
        dy = grid_y - robot_y  # [B,H,W]
        dist_sq = dx * dx + dy * dy

        # 5) footprint 半径内的都清空为 free=1
        r2 = radius * radius
        mask = dist_sq <= r2  # [B,H,W] bool

        occ_new = occupancy.clone()
        # 0=free, 1=occupied
        occ_new[mask] = 0.0

        return occ_new

    @torch.no_grad()
    def dilate_occupancy(
        self,
        occ: torch.Tensor,
        radius: float,
        resolution: float,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        对占用栅格进行"机器人半径"膨胀，得到 inflated costmap（只做 0/1 膨胀版）。
        支持批量并行处理多个环境的 occupancy maps。

        参数：
            occ: [H, W] 或 [1, H, W] 或 [B, H, W] 或 [B, 1, H, W]，0=空闲，>0=占用
            radius: 膨胀半径（米）
            resolution: 栅格分辨率（米/格）
            device: torch设备（如果为None，则使用occ的设备）

        返回：
            dilated_occ: 与输入形状对应的膨胀后的占用栅格
                - 输入 [H, W] -> 输出 [H, W]
                - 输入 [B, H, W] -> 输出 [B, H, W]
        """
        # 转换 numpy array 到 torch tensor
        if isinstance(occ, np.ndarray):
            occ = torch.from_numpy(occ)

        # 确保在正确的设备上
        if device is None:
            device = occ.device if hasattr(occ, "device") else self.device
        occ = occ.to(device)

        # 记录原始形状以便后续恢复
        original_shape = occ.shape
        original_dim = occ.dim()

        # 标准化输入形状到 [B, 1, H, W] 格式
        if occ.dim() == 2:
            # 单张图: [H, W] -> [1, 1, H, W]
            occ4d = occ.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif occ.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            occ4d = occ.unsqueeze(1)  # [B, 1, H, W]
        elif occ.dim() == 4:
            # [B, 1, H, W] 或 [B, C, H, W]，直接使用
            occ4d = occ
        else:
            raise ValueError(f"occ dim must be 2, 3, or 4, got {occ.shape}")

        B, C, H, W = occ4d.shape
        # 如果有多个通道，合并所有通道（任一通道占用则为占用）
        if C > 1:
            occ4d = occ4d.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]

        # 1) 计算膨胀半径对应的 cell 半径
        radius_cells = math.ceil(radius / resolution)
        if radius_cells <= 0:
            # 半径太小，直接返回原图
            return occ.clone()

        # 2) 构造圆形结构元素 kernel: [1, 1, K, K]
        #    所有batch共享同一个kernel，所以只需要构造一次
        ksize = 2 * radius_cells + 1
        ys, xs = torch.meshgrid(
            torch.arange(ksize, device=device, dtype=torch.float32),
            torch.arange(ksize, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cy, cx = float(radius_cells), float(radius_cells)
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
        # 圆形掩码：距离中心 <= r 的格子为 1
        kernel = (dist2 <= radius_cells**2).float()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

        # 3) 用 conv2d 做"膨胀"：只要邻域有一个占用，则输出为占用
        #    这里假设 occ>0 的位置为占用，所以先把 occ>0 转成 0/1
        #    批量处理：对所有 B 个环境并行执行卷积
        occ_bin = (occ4d > 0).float()  # [B, 1, H, W]

        # 对每个batch并行执行卷积
        # groups=B 可以实现对每个样本使用不同的kernel，但这里所有样本用同一个kernel
        # 所以使用标准的conv2d即可，它会自动对每个batch并行处理
        conv = F.conv2d(
            occ_bin,  # [B, 1, H, W]
            kernel,  # [1, 1, K, K]
            padding=radius_cells,
            groups=1,  # 标准卷积，对batch中每个样本并行处理
        )  # [B, 1, H, W]

        dilated = (conv > 0).to(occ.dtype)  # [B, 1, H, W]

        # 4) 恢复原始形状
        if original_dim == 2:
            # 单张图: [1, 1, H, W] -> [H, W]
            dilated = dilated.squeeze(0).squeeze(0)  # [H, W]
        elif original_dim == 3:
            # [B, H, W] -> [B, H, W]
            dilated = dilated.squeeze(1)  # [B, H, W]
        elif original_dim == 4:
            # [B, 1, H, W] 或 [B, C, H, W]
            if original_shape[1] == 1:
                pass  # 已经是 [B, 1, H, W]，保持不变
            else:
                # 原始有多个通道，现在只有1个，需要恢复到原始通道数
                dilated = dilated.expand(-1, original_shape[1], -1, -1)  # [B, C, H, W]

        return dilated

    @torch.no_grad()
    def rollout_unicycle(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
        controls_per_env: list,
        occupancy: torch.Tensor,  # [B, H, W]
        num_steps: int = 20,
    ):
        """
        差分驱动版 rollout：
        - controls_per_env[b][k] = (v, omega)
        - 运动学: x_{t+1} = x_t + v cos(theta) dt
                y_{t+1} = y_t + v sin(theta) dt
                theta_{t+1} = theta_t + omega dt
        其它（costmap 一致、碰撞检测一致）
        """
        device = self.device
        env_ids = env_ids.to(device)
        occupancy = occupancy.to(device)
        # print("robot_state: ", robot_state)
        base_pos = robot_state["base_pos"].to(device)  # [N_env, 3]
        base_quat = robot_state["base_quat"].to(device)  # [N_env, 4]

        B = env_ids.shape[0]
        dt = self.dt

        H, W = occupancy.shape[-2], occupancy.shape[-1]

        num_candidates_per_env = [c.shape[0] for c in controls_per_env]
        max_Nc = max(num_candidates_per_env) if num_candidates_per_env else 0
        if max_Nc == 0:
            return [], []

        map_metas = self.map_meta_local_all[env_ids]  # [B, 3]
        origin_x = map_metas[:, 0:1]  # [B,1]
        origin_y = map_metas[:, 1:2]  # [B,1]
        resolution = map_metas[:, 2:3]  # 如果你将来想支持 per-env resolution

        # yaw from quat
        # print("base_quat: ", base_quat)
        # print("env_ids: ", env_ids)
        if base_quat.shape == (B, 4):
            qw = base_quat[env_ids, 0]
            qx = base_quat[env_ids, 1]
            qy = base_quat[env_ids, 2]
            qz = base_quat[env_ids, 3]
            yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )  # [B]
        else:
            yaw = base_quat[env_ids].squeeze(-1)
        # print("yaw: ", yaw)

        start_states = torch.stack(
            [
                base_pos[env_ids, 0],
                base_pos[env_ids, 1],
                yaw,
            ],
            dim=-1,
        )  # [B, 3]

        # pad controls
        controls_padded = []
        valid_mask_list = []
        for b in range(B):
            controls_b = controls_per_env[b].to(device)  # [Nc_b, 2] (v, omega)
            Nc_b = controls_b.shape[0]
            if Nc_b < max_Nc:
                padding = torch.zeros(
                    max_Nc - Nc_b, 2, dtype=controls_b.dtype, device=device
                )
                controls_padded_b = torch.cat([controls_b, padding], dim=0)
                valid_mask_b = torch.cat(
                    [
                        torch.ones(Nc_b, dtype=torch.bool, device=device),
                        torch.zeros(max_Nc - Nc_b, dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            else:
                controls_padded_b = controls_b
                valid_mask_b = torch.ones(max_Nc, dtype=torch.bool, device=device)

            controls_padded.append(controls_padded_b)
            valid_mask_list.append(valid_mask_b)

        all_controls = torch.stack(controls_padded, dim=0)  # [B, max_Nc, 2]
        valid_mask = torch.stack(valid_mask_list, dim=0)  # [B, max_Nc]

        v = all_controls[:, :, 0]  # [B, max_Nc]
        omega = all_controls[:, :, 1]  # [B, max_Nc]

        trajectories = torch.zeros(
            B, max_Nc, num_steps, 3, dtype=torch.float32, device=device
        )

        start_states_expanded = start_states.unsqueeze(1).expand(-1, max_Nc, -1)
        current_x = start_states_expanded[:, :, 0]
        current_y = start_states_expanded[:, :, 1]
        current_yaw = start_states_expanded[:, :, 2]

        collided = torch.zeros(B, max_Nc, dtype=torch.bool, device=device)

        for t in range(num_steps):
            trajectories[:, :, t, 0] = current_x
            trajectories[:, :, t, 1] = current_y
            trajectories[:, :, t, 2] = current_yaw

            active = ~collided & valid_mask

            grid_j = (current_x - origin_x) / resolution
            grid_i = (current_y - origin_y) / resolution

            grid_j = torch.round(grid_j).long()
            grid_i = torch.round(grid_i).long()

            valid_x = (grid_j >= 0) & (grid_j < W)
            valid_y = (grid_i >= 0) & (grid_i < H)
            in_bounds = valid_x & valid_y & active

            grid_j_clamped = torch.clamp(grid_j, 0, W - 1)
            grid_i_clamped = torch.clamp(grid_i, 0, H - 1)

            batch_indices = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, max_Nc)
            )
            occupancy_values = occupancy[batch_indices, grid_i_clamped, grid_j_clamped]

            hit_obstacle = (occupancy_values > 0.5) & in_bounds
            out_of_bounds = (~in_bounds) & active
            new_collisions = hit_obstacle | out_of_bounds
            # debug，仅看 env0：
            # env0 = 0
            # if (out_of_bounds[env0] & ~collided[env0]).any():
            #     bad = torch.nonzero(out_of_bounds[env0] & ~collided[env0], as_tuple=False).squeeze(-1)
            #     for k in bad.tolist():
            #         print(
            #             f"[out_of_bounds] env0 cand={k}, t={t}, "
            #             f"x={current_x[env0,k].item():+.4f}, y={current_y[env0,k].item():+.4f}"
            #         )
            collided = collided | new_collisions

            if collided.all():
                if t < num_steps - 1:
                    for t_remain in range(t + 1, num_steps):
                        trajectories[:, :, t_remain, 0] = current_x
                        trajectories[:, :, t_remain, 1] = current_y
                        trajectories[:, :, t_remain, 2] = current_yaw
                break

            update_mask = active & ~collided

            # ✅ 差分驱动 / unicycle 运动学
            dx = v * torch.cos(current_yaw) * dt
            dy = v * torch.sin(current_yaw) * dt
            dtheta = omega * dt

            current_x = torch.where(update_mask, current_x + dx, current_x)
            current_y = torch.where(update_mask, current_y + dy, current_y)
            current_yaw = torch.where(update_mask, current_yaw + dtheta, current_yaw)

            current_yaw = torch.atan2(torch.sin(current_yaw), torch.cos(current_yaw))

        trajectories_per_env = []
        collision_masks_per_env = []
        for b in range(B):
            Nc_b = num_candidates_per_env[b]
            trajs_b = trajectories[b, :Nc_b, :, :]
            collided_b = collided[b, :Nc_b]
            trajectories_per_env.append(trajs_b)
            collision_masks_per_env.append(collided_b)

        return trajectories_per_env, collision_masks_per_env

    @torch.no_grad()
    def rollout_holonomic(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
        controls_per_env: list,
        occupancy: torch.Tensor,  # [B, H, W]
        num_steps: int = 20,
    ):
        """
        全向 / holonomic 底盘版 rollout：
        - controls_per_env[b][k] = (vx, vy, rz)
        - 运动学（机器人坐标系 → 世界坐标系）:
            dx_body = vx * dt
            dy_body = vy * dt
            dx_world = dx_body * cos(theta) - dy_body * sin(theta)
            dy_world = dx_body * sin(theta) + dy_body * cos(theta)
            theta_{t+1} = theta_t + rz * dt
        其它（costmap、碰撞检测）与 unicycle 版保持一致。
        """
        device = self.device
        env_ids = env_ids.to(device)
        occupancy = occupancy.to(device)

        base_pos = robot_state["base_pos"].to(device)  # [N_env, 3]
        base_quat = robot_state["base_quat"].to(device)  # [N_env, 4] 或 [N_env,1]

        B = env_ids.shape[0]
        dt = self.dt

        H, W = occupancy.shape[-2], occupancy.shape[-1]

        # 每个 env 候选控制数
        num_candidates_per_env = [c.shape[0] for c in controls_per_env]
        max_Nc = max(num_candidates_per_env) if num_candidates_per_env else 0
        if max_Nc == 0:
            return [], []

        # 每个 env 对应的 costmap meta
        map_metas = self.map_meta_local_all[env_ids]  # [B, 3]
        origin_x = map_metas[:, 0:1]  # [B,1]
        origin_y = map_metas[:, 1:2]  # [B,1]
        resolution = map_metas[:, 2:3]  # [B,1]，如后续要 per-env 分辨率

        # -------- yaw from quat or joint pos --------
        if base_quat.shape[1] == 4:
            # [N_env,4] -> 用真正的四元数算 yaw
            qw = base_quat[env_ids, 0]
            qx = base_quat[env_ids, 1]
            qy = base_quat[env_ids, 2]
            qz = base_quat[env_ids, 3]
            yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )  # [B]
        else:
            # holonomic base: base_quat 实际上是 joint_pos[:, dummy_base_revolute_z_joint]
            yaw = base_quat[env_ids].squeeze(-1)  # [B]
        # print("yaw: ", yaw)
        # 初始状态 [x, y, yaw]
        start_states = torch.stack(
            [
                base_pos[env_ids, 0],
                base_pos[env_ids, 1],
                yaw,
            ],
            dim=-1,
        )  # [B, 3]

        # ---------- pad controls 到统一长度 ----------
        controls_padded = []
        valid_mask_list = []
        for b in range(B):
            controls_b = controls_per_env[b].to(device)  # [Nc_b, 3] (vx, vy, rz)
            Nc_b = controls_b.shape[0]
            if Nc_b < max_Nc:
                padding = torch.zeros(
                    max_Nc - Nc_b, 3, dtype=controls_b.dtype, device=device
                )
                controls_padded_b = torch.cat([controls_b, padding], dim=0)
                valid_mask_b = torch.cat(
                    [
                        torch.ones(Nc_b, dtype=torch.bool, device=device),
                        torch.zeros(max_Nc - Nc_b, dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            else:
                controls_padded_b = controls_b
                valid_mask_b = torch.ones(max_Nc, dtype=torch.bool, device=device)

            controls_padded.append(controls_padded_b)
            valid_mask_list.append(valid_mask_b)

        all_controls = torch.stack(controls_padded, dim=0)  # [B, max_Nc, 3]
        valid_mask = torch.stack(valid_mask_list, dim=0)  # [B, max_Nc]

        vx = all_controls[:, :, 0]  # [B, max_Nc]
        vy = all_controls[:, :, 1]  # [B, max_Nc]
        rz = all_controls[:, :, 2]  # [B, max_Nc]

        # 轨迹缓存
        trajectories = torch.zeros(
            B, max_Nc, num_steps, 3, dtype=torch.float32, device=device
        )

        start_states_expanded = start_states.unsqueeze(1).expand(-1, max_Nc, -1)
        current_x = start_states_expanded[:, :, 0]  # [B, max_Nc]
        current_y = start_states_expanded[:, :, 1]  # [B, max_Nc]
        current_yaw = start_states_expanded[:, :, 2]  # [B, max_Nc]

        collided = torch.zeros(B, max_Nc, dtype=torch.bool, device=device)

        for t in range(num_steps):
            # 记录当前 pose
            trajectories[:, :, t, 0] = current_x
            trajectories[:, :, t, 1] = current_y
            trajectories[:, :, t, 2] = current_yaw

            active = ~collided & valid_mask  # 仍在模拟的轨迹

            # ---- 投影到栅格坐标系 ----
            grid_j = (current_x - origin_x) / resolution  # x -> col
            grid_i = (current_y - origin_y) / resolution  # y -> row

            grid_j = torch.round(grid_j).long()
            grid_i = torch.round(grid_i).long()

            valid_x = (grid_j >= 0) & (grid_j < W)
            valid_y = (grid_i >= 0) & (grid_i < H)
            in_bounds = valid_x & valid_y & active

            grid_j_clamped = torch.clamp(grid_j, 0, W - 1)
            grid_i_clamped = torch.clamp(grid_i, 0, H - 1)

            batch_indices = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, max_Nc)
            )
            occupancy_values = occupancy[batch_indices, grid_i_clamped, grid_j_clamped]

            hit_obstacle = (occupancy_values > 0.5) & in_bounds
            out_of_bounds = (~in_bounds) & active
            new_collisions = hit_obstacle | out_of_bounds

            collided = collided | new_collisions

            # 全部撞了 / 出界了，后面时间步 pose 保持不变
            if collided.all():
                if t < num_steps - 1:
                    for t_remain in range(t + 1, num_steps):
                        trajectories[:, :, t_remain, 0] = current_x
                        trajectories[:, :, t_remain, 1] = current_y
                        trajectories[:, :, t_remain, 2] = current_yaw
                break

            update_mask = active & ~collided

            # ====== Holonomic 运动学（body → world）======
            cos_yaw = torch.cos(current_yaw)
            sin_yaw = torch.sin(current_yaw)

            # body frame 位移
            dx_body = vx * dt
            dy_body = vy * dt
            dtheta = rz * dt

            # 转到 world frame
            dx_world = dx_body * cos_yaw - dy_body * sin_yaw
            dy_world = dx_body * sin_yaw + dy_body * cos_yaw

            current_x = torch.where(update_mask, current_x + dx_world, current_x)
            current_y = torch.where(update_mask, current_y + dy_world, current_y)
            current_yaw = torch.where(update_mask, current_yaw + dtheta, current_yaw)

            # 归一化角度到 [-pi, pi]
            current_yaw = torch.atan2(torch.sin(current_yaw), torch.cos(current_yaw))

        # 拆回 per-env 列表
        trajectories_per_env = []
        collision_masks_per_env = []
        for b in range(B):
            Nc_b = num_candidates_per_env[b]
            trajs_b = trajectories[b, :Nc_b, :, :]  # [Nc_b, T, 3]
            collided_b = collided[b, :Nc_b]  # [Nc_b]
            trajectories_per_env.append(trajs_b)
            collision_masks_per_env.append(collided_b)

        return trajectories_per_env, collision_masks_per_env

    @torch.no_grad()
    def rollout(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
        controls_per_env: list,
        occupancy: torch.Tensor,  # [B, H, W] 膨胀后占用图
        num_steps: int = 20,
    ):
        """
        对每个 env、每个 candidate 控制(v, steer) 做 rollout（完全并行化）。

        输入:
            robot_state:  Isaac 的 robot_state 字典
            env_ids:      [B] tensor，表示这次参与规划的 env 索引
            controls_per_env: List[Tensor]，len = B，controls_per_env[b]: [Nc_b, 2]
            occupancy:    [B, H, W] 的膨胀占用图（0/1）
            num_steps:    rollout 步数

        返回:
            trajectories_per_env: List[Tensor]，len=B，
                trajectories_per_env[b]: [Nc_b, num_steps, 3]
            collision_masks_per_env: List[BoolTensor]，len=B，
                collision_masks_per_env[b]: [Nc_b]，True 表示该控制撞了
        """
        device = self.device
        env_ids = env_ids.to(device)
        occupancy = occupancy.to(device)

        base_pos = robot_state["base_pos"].to(device)  # [N_env, 3]
        base_quat = robot_state["base_quat"].to(device)  # [N_env, 4]

        B = env_ids.shape[0]
        dt = self.dt
        wheel_base = self.wheel_base

        # 获取占用地图尺寸
        H, W = occupancy.shape[-2], occupancy.shape[-1]

        # 找到每个环境的候选数量，以及最大候选数量
        num_candidates_per_env = [c.shape[0] for c in controls_per_env]
        max_Nc = max(num_candidates_per_env) if num_candidates_per_env else 0

        if max_Nc == 0:
            return [], []

        map_metas = self.map_meta_local_all[env_ids]  # [B, 3]
        origin_x = map_metas[:, 0:1]  # [B,1]
        origin_y = map_metas[:, 1:2]  # [B,1]
        resolution = map_metas[:, 2:3]  # 如果你将来想支持 per-env resolution

        # ----------------------------------------
        # 1) 准备初始状态和候选控制（向量化）
        # ----------------------------------------
        # 从四元数计算 yaw 角
        qw = base_quat[env_ids, 0]  # [B]
        qx = base_quat[env_ids, 1]  # [B]
        qy = base_quat[env_ids, 2]  # [B]
        qz = base_quat[env_ids, 3]  # [B]

        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )  # [B]

        # 初始状态: [B, 3]
        start_states = torch.stack(
            [
                base_pos[env_ids, 0],  # x
                base_pos[env_ids, 1],  # y
                yaw,  # theta
            ],
            dim=-1,
        )  # [B, 3]

        # 将所有候选控制 pad 到相同大小
        # 创建 mask 标记有效候选
        controls_padded = []
        valid_mask_list = []
        for b in range(B):
            controls_b = controls_per_env[b].to(device)  # [Nc_b, 2]
            Nc_b = controls_b.shape[0]

            if Nc_b < max_Nc:
                # Pad 到 max_Nc
                padding = torch.zeros(
                    max_Nc - Nc_b, 2, dtype=controls_b.dtype, device=device
                )
                controls_padded_b = torch.cat(
                    [controls_b, padding], dim=0
                )  # [max_Nc, 2]
                valid_mask_b = torch.cat(
                    [
                        torch.ones(Nc_b, dtype=torch.bool, device=device),
                        torch.zeros(max_Nc - Nc_b, dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )  # [max_Nc]
            else:
                controls_padded_b = controls_b  # [max_Nc, 2]
                valid_mask_b = torch.ones(
                    max_Nc, dtype=torch.bool, device=device
                )  # [max_Nc]

            controls_padded.append(controls_padded_b)
            valid_mask_list.append(valid_mask_b)

        # 堆叠成 tensor: [B, max_Nc, 2]
        all_controls = torch.stack(controls_padded, dim=0)  # [B, max_Nc, 2]
        valid_mask = torch.stack(valid_mask_list, dim=0)  # [B, max_Nc]

        # 提取速度和转向角: [B, max_Nc]
        v = all_controls[:, :, 0]  # [B, max_Nc]
        steer = all_controls[:, :, 1]  # [B, max_Nc]

        # ----------------------------------------
        # 2) 向量化轨迹预测（阿克曼模型）
        # ----------------------------------------
        # 初始化轨迹: [B, max_Nc, num_steps, 3]
        trajectories = torch.zeros(
            B, max_Nc, num_steps, 3, dtype=torch.float32, device=device
        )

        # 扩展初始状态: [B, 3] -> [B, max_Nc, 3]
        # start_states: [B, 3] -> [B, 1, 3] -> [B, max_Nc, 3]
        start_states_expanded = start_states.unsqueeze(1).expand(
            -1, max_Nc, -1
        )  # [B, max_Nc, 3]
        current_x = start_states_expanded[:, :, 0]  # [B, max_Nc]
        current_y = start_states_expanded[:, :, 1]  # [B, max_Nc]
        current_yaw = start_states_expanded[:, :, 2]  # [B, max_Nc]

        # 碰撞标记: [B, max_Nc]
        collided = torch.zeros(B, max_Nc, dtype=torch.bool, device=device)

        # 逐时间步预测（时间维度需要循环，但所有候选并行）
        for t in range(num_steps):
            # 记录当前状态
            trajectories[:, :, t, 0] = current_x
            trajectories[:, :, t, 1] = current_y
            trajectories[:, :, t, 2] = current_yaw

            # 对于已经碰撞的轨迹，不再更新（保持当前位置）
            active = ~collided & valid_mask  # [B, max_Nc]

            # 将轨迹点映射到网格坐标
            # current_x: [B, max_Nc], origin_x: [B, 1]
            # 需要broadcasting: [B, max_Nc] - [B, 1] -> [B, max_Nc]
            grid_j = (current_x - origin_x) / resolution  # [B, max_Nc]
            grid_i = (current_y - origin_y) / resolution  # [B, max_Nc]

            grid_j = torch.round(grid_j).long()  # [B, max_Nc]
            grid_i = torch.round(grid_i).long()  # [B, max_Nc]

            # 检查边界
            valid_x = (grid_j >= 0) & (grid_j < W)  # [B, max_Nc]
            valid_y = (grid_i >= 0) & (grid_i < H)  # [B, max_Nc]
            in_bounds = valid_x & valid_y & active  # [B, max_Nc]

            # 将越界坐标 clamp 到有效范围（避免索引错误）
            grid_j_clamped = torch.clamp(grid_j, 0, W - 1)
            grid_i_clamped = torch.clamp(grid_i, 0, H - 1)

            # 批量查询占用值
            batch_indices = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, max_Nc)
            )  # [B, max_Nc]
            occupancy_values = occupancy[
                batch_indices, grid_i_clamped, grid_j_clamped
            ]  # [B, max_Nc]

            # 碰撞检测：占用值 > 0.5 或越界
            hit_obstacle = (occupancy_values > 0.5) & in_bounds  # [B, max_Nc]
            out_of_bounds = (~in_bounds) & active  # [B, max_Nc]
            new_collisions = hit_obstacle | out_of_bounds  # [B, max_Nc]

            # 更新碰撞标记
            collided = collided | new_collisions  # [B, max_Nc]

            # 如果所有轨迹都碰撞了，提前退出
            if collided.all():
                # 填充剩余时间步
                if t < num_steps - 1:
                    for t_remain in range(t + 1, num_steps):
                        trajectories[:, :, t_remain, 0] = current_x
                        trajectories[:, :, t_remain, 1] = current_y
                        trajectories[:, :, t_remain, 2] = current_yaw
                break

            # 仅对未碰撞且有效的轨迹进行状态更新
            update_mask = active & ~collided  # [B, max_Nc]

            # 阿克曼模型更新（仅更新未碰撞的）
            # x_{t+1} = x_t + v cos(yaw) dt
            # y_{t+1} = y_t + v sin(yaw) dt
            # yaw_{t+1} = yaw_t + v/L * tan(steer) dt
            dx = v * torch.cos(current_yaw) * dt  # [B, max_Nc]
            dy = v * torch.sin(current_yaw) * dt  # [B, max_Nc]
            dtheta = (v / wheel_base) * torch.tan(steer) * dt  # [B, max_Nc]

            # 仅更新未碰撞的轨迹
            current_x = torch.where(
                update_mask, current_x + dx, current_x
            )  # [B, max_Nc]
            current_y = torch.where(
                update_mask, current_y + dy, current_y
            )  # [B, max_Nc]
            current_yaw = torch.where(
                update_mask, current_yaw + dtheta, current_yaw
            )  # [B, max_Nc]

            # 归一化角度到 [-pi, pi]
            current_yaw = torch.atan2(
                torch.sin(current_yaw), torch.cos(current_yaw)
            )  # [B, max_Nc]

        # ----------------------------------------
        # 3) 分割结果并返回每个环境的轨迹
        # ----------------------------------------
        trajectories_per_env = []
        collision_masks_per_env = []

        for b in range(B):
            Nc_b = num_candidates_per_env[b]
            trajs_b = trajectories[b, :Nc_b, :, :]  # [Nc_b, num_steps, 3]
            collided_b = collided[b, :Nc_b]  # [Nc_b]

            trajectories_per_env.append(trajs_b)
            collision_masks_per_env.append(collided_b)

        return trajectories_per_env, collision_masks_per_env

    def evaluate_and_select_actions(
        self,
        trajs_list,  # List[Tensor], len=B，每个 [Nc, T, 3]
        collided_list,  # List[BoolTensor], len=B，每个 [Nc]
        candidates: torch.Tensor,  # [B, Nc, 2]，所有 env 的 (v, steer)
        target_pos: torch.Tensor,  # [B, Tp, 3]，每个 env 的路径点 (x,y,heading)
    ) -> torch.Tensor:
        """
        DWB + 终点朝向 P 控制：
        - 离终点较远：正常 DWB 选 (v, steer)
        - 离终点足够近：不再规划，直接 v=0，只用 P 控制转向对齐目标 heading
        """

        device = self.device

        # 1) 堆叠 rollout 结果
        trajs = torch.stack(trajs_list, dim=0).to(device)  # [B, Nc, T, 3]
        collided = torch.stack(collided_list, dim=0).to(device)  # [B, Nc] (bool)
        candidates = candidates.to(device)  # [B, Nc, 2]

        B, Nc, T, _ = trajs.shape

        # 2) 处理 target_pos: [B, Tp, 3]
        target_pos = target_pos.to(device)
        assert target_pos.dim() == 3 and target_pos.shape[0] == B, (
            f"target_pos 应该是 [B, Tp, 3]，但拿到的是 {target_pos.shape}"
        )

        waypoints_xy = target_pos[..., 0:2]  # [B, Tp, 2]
        waypoints_yaw = target_pos[..., 2]  # [B, Tp]
        Tp = waypoints_xy.shape[1]

        # 3) 取每条轨迹的末端点 (x,y) 以及起始位姿
        final_xy = trajs[:, :, -1, 0:2]  # [B, Nc, 2]
        base_xy = trajs[:, 0, 0, 0:2]  # [B, 2] 当前实际机器人位置
        base_yaw = trajs[:, 0, 0, 2]  # [B]   当前机器人朝向

        # ======== A) 正常 DWB 代价计算：path_cost / goal_cost / obs / vel ========

        # --- 当前路径进度 idx0 ---
        diff_base = waypoints_xy - base_xy.unsqueeze(1)  # [B, Tp, 2]
        dist2_base = torch.sum(diff_base * diff_base, dim=-1)  # [B, Tp]
        idx0 = torch.argmin(dist2_base, dim=-1)  # [B]

        idxs = torch.arange(Tp, device=device).unsqueeze(0).expand(B, -1)  # [B, Tp]
        forward_mask = idxs >= idx0.unsqueeze(1)  # [B, Tp]

        # --- 末端点到前方路径点的距离 ---
        final_xy_exp = final_xy.unsqueeze(2)  # [B, Nc, 1, 2]
        waypoints_exp = waypoints_xy.unsqueeze(1)  # [B, 1, Tp, 2]
        delta_all = final_xy_exp - waypoints_exp  # [B, Nc, Tp, 2]
        dists = torch.linalg.norm(delta_all, dim=-1)  # [B, Nc, Tp]

        BIG = 1e6
        forward_mask_expanded = forward_mask.unsqueeze(1)  # [B, 1, Tp]
        dists_forward = torch.where(
            forward_mask_expanded,
            dists,
            torch.full_like(dists, BIG),
        )  # [B, Nc, Tp]

        path_cost = dists_forward.min(dim=-1).values  # [B, Nc]

        # --- 终点位置距离 ---
        goal_xy = waypoints_xy[:, -1, :]  # [B, 2]
        delta_goal = final_xy - goal_xy.unsqueeze(1)  # [B, Nc, 2]
        goal_cost = torch.linalg.norm(delta_goal, dim=-1)  # [B, Nc]

        # --- 碰撞代价 ---
        big_penalty = 1e4
        obstacle_cost = torch.where(
            collided,
            torch.full_like(goal_cost, big_penalty),
            torch.zeros_like(goal_cost),
        )  # [B, Nc]

        # --- 速度代价（鼓励往前）---
        v = candidates[..., 0]  # [B, Nc]
        vel_cost = -v  # [B, Nc]

        total_cost = (
            self.w_path * path_cost
            + self.w_goal * goal_cost
            + self.w_obs * obstacle_cost
            + self.w_vel * vel_cost
        )  # [B, Nc]

        # ======== B) 终点附近：用 P 控制只转向 ========

        # 当前到终点的距离（用 base_xy）
        dist_to_goal_now = torch.linalg.norm(base_xy - goal_xy, dim=-1)  # [B]

        # 终点位置阈值：比如 0.2m 内认为到了“终点附近”
        goal_pos_thresh = getattr(self, "goal_pos_threshold", 0.2)
        near_goal_mask = dist_to_goal_now < goal_pos_thresh  # [B] bool

        goal_yaw = waypoints_yaw[:, -1]  # [B]
        yaw_err_now = angle_diff(goal_yaw, base_yaw)  # [B]

        # 当 |yaw_err| 大于一定阈值时，我们希望原地转向对齐
        yaw_align_thresh = getattr(
            self, "goal_yaw_threshold", 5.0 * math.pi / 180.0
        )  # ~5°
        # 需要执行“原地转向”的 env
        turn_only_mask = near_goal_mask & (yaw_err_now.abs() > yaw_align_thresh)  # [B]

        # 最终输出 action
        actions = torch.zeros(B, 2, device=device)  # [B, 2] = (v, steer)

        # ---- (B1) 对于需要 turn_only 的 env：只发 (0, steer) ----
        if turn_only_mask.any():
            idx_turn = torch.nonzero(turn_only_mask, as_tuple=False).squeeze(-1)  # [Nt]
            # print("yaw_err_now: ", yaw_err_now)
            # P 控制：steer = k_yaw * yaw_err（正负号自然由 yaw_err_now 决定转向方向）
            k_yaw = getattr(self, "k_goal_yaw", 1.0)
            steer_cmd = k_yaw * yaw_err_now[idx_turn]  # [Nt]

            # 限制在允许的角速度范围内（你可以用 min/max_yaw_rate 或单独的 goal_max_yaw_rate）
            max_steer = getattr(self, "goal_max_yaw_rate", self.max_yaw_rate)
            steer_cmd = torch.clamp(steer_cmd, -max_steer, max_steer)
            # print("steer_cmd: ", steer_cmd)
            actions[idx_turn, 0] = 0.0  # v = 0 原地转向
            actions[idx_turn, 1] = steer_cmd  # steer = P 控制出来的角速度

        # ---- (B2) 对于其他 env：正常 DWB 选最优 (v, steer) ----
        follow_mask = ~turn_only_mask
        if follow_mask.any():
            idx_follow = torch.nonzero(follow_mask, as_tuple=False).squeeze(-1)  # [Nf]

            total_cost_follow = total_cost[follow_mask]  # [Nf, Nc]
            best_idx_follow = torch.argmin(total_cost_follow, dim=1)  # [Nf]

            env_idx = idx_follow
            cand_idx = best_idx_follow
            actions[env_idx, :] = candidates[env_idx, cand_idx, :]  # [Nf, 2]

        # Debug 信息
        debug_info = {
            "trajs": trajs,
            "candidates": candidates,
            "total_cost": total_cost,
            "path_cost": path_cost,
            "goal_cost": goal_cost,
            "obstacle_cost": obstacle_cost,
            "vel_cost": vel_cost,
            "best_idx_global": torch.argmin(total_cost, dim=1),
            "dist_to_goal_now": dist_to_goal_now,
            "yaw_err_now": yaw_err_now,
            "near_goal_mask": near_goal_mask,
            "turn_only_mask": turn_only_mask,
        }
        return actions, debug_info

    def evaluate_and_select_actions_holonomic(
        self,
        trajs_list,  # List[Tensor], len=B，每个 [Nc, T, 3]
        collided_list,  # List[BoolTensor], len=B，每个 [Nc]
        candidates: torch.Tensor,  # [B, Nc, 3]，所有 env 的 (vx, vy, rz)
        target_pos: torch.Tensor,  # [B, Tp, 2]，每个 env 的路径点
    ) -> torch.Tensor:
        """
        Holonomic 版本 DWB 评估：
        - 对每个 env 的所有 (vx, vy, rz) 候选进行打分
        - 使用路径 [B, Tp, 2] 作为 reference：
            * path_cost: 末端点到“当前车前方路径”的最近距离
            * goal_cost: 末端点到终点的距离
            * obstacle_cost: 碰撞罚分
            * vel_cost: 使用 speed = sqrt(vx^2 + vy^2) 鼓励更快运动
        - 返回每个 env 的最佳 (vx, vy, rz): [B, 3]
        """
        # device = self.device
        trajs = torch.stack(trajs_list, dim=0).to(self.device)  # [B, Nc, T, 3]
        collided = torch.stack(collided_list, dim=0).to(self.device)  # [B, Nc]
        candidates = candidates.to(self.device)  # [B, Nc, 3]

        B, Nc, T, _ = trajs.shape

        target_pos = target_pos.to(self.device)
        assert (
            target_pos.dim() == 3
            and target_pos.shape[0] == B
            and target_pos.shape[2] >= 3
        ), f"target_pos 应该是 [B, Tp, 3]，但拿到的是 {target_pos.shape}"

        waypoints_xy = target_pos[..., 0:2]  # [B, Tp, 2]
        waypoints_yaw = target_pos[..., 2]  # [B, Tp]
        Tp = waypoints_xy.shape[1]

        final_xy = trajs[:, :, -1, 0:2]  # [B, Nc, 2]
        base_xy = trajs[:, 0, 0, 0:2]  # [B, 2]
        base_yaw = trajs[:, 0, 0, 2]  # [B]

        # ===== A) 正常 DWB cost =====
        diff_base = waypoints_xy - base_xy.unsqueeze(1)  # [B, Tp, 2]
        dist2_base = torch.sum(diff_base * diff_base, dim=-1)  # [B, Tp]
        idx0 = torch.argmin(dist2_base, dim=-1)  # [B]

        idxs = torch.arange(Tp, device=self.device).unsqueeze(0).expand(B, -1)
        forward_mask = idxs >= idx0.unsqueeze(1)  # [B, Tp]

        # path_cost: 整条轨迹每个时间步到最近前向路径点的平均位姿误差
        # 位姿误差 = xy 距离 + |yaw 差|，统一由 w_path 缩放
        # 每步处理一个时间步避免 [B, Nc, T, Tp] 超大 tensor
        BIG = 1e6
        fw_mask_nc = forward_mask.unsqueeze(1)  # [B, 1, Tp]
        waypoints_nc = waypoints_xy.unsqueeze(1)  # [B, 1, Tp, 2]
        waypoints_yaw_nc = waypoints_yaw.unsqueeze(1)  # [B, 1, Tp]
        path_cost_acc = torch.zeros(B, Nc, device=self.device)
        stride = max(1, T // 20)  # 最多取 20 个时间步，加快速度
        sampled_steps = range(0, T, stride)
        for t in sampled_steps:
            traj_t_xy = trajs[:, :, t, 0:2]  # [B, Nc, 2]
            delta_t = traj_t_xy.unsqueeze(2) - waypoints_nc  # [B, Nc, Tp, 2]
            dists_t = torch.linalg.norm(delta_t, dim=-1)  # [B, Nc, Tp]
            dists_fwd_t = torch.where(
                fw_mask_nc,
                dists_t,
                torch.full_like(dists_t, BIG),
            )
            min_xy_dists, nearest_wp_idx = dists_fwd_t.min(dim=-1)  # [B, Nc]
            traj_t_yaw = trajs[:, :, t, 2]  # [B, Nc]
            nearest_wp_yaw = (
                waypoints_yaw_nc.expand(-1, Nc, -1)
                .gather(2, nearest_wp_idx.unsqueeze(2))
                .squeeze(2)
            )  # [B, Nc]
            yaw_err_t = angle_diff(traj_t_yaw, nearest_wp_yaw).abs()  # [B, Nc]
            path_cost_acc += min_xy_dists + yaw_err_t
        path_cost = path_cost_acc / len(sampled_steps)

        goal_xy = waypoints_xy[:, -1, :]  # [B, 2]
        delta_goal = final_xy - goal_xy.unsqueeze(1)  # [B, Nc, 2]
        goal_cost = torch.linalg.norm(delta_goal, dim=-1)  # [B, Nc]

        big_penalty = 1e4
        obstacle_cost = torch.where(
            collided,
            torch.full_like(goal_cost, big_penalty),
            torch.zeros_like(goal_cost),
        )

        vx = candidates[..., 0]  # 向前速度
        vel_cost = -vx  # 速度代价只考虑 vx，鼓励向前移动

        vy = candidates[..., 1]  # 横向速度
        vy_cost = vy.abs()  # 惩罚蟹行，|vy| 越大代价越高

        total_cost = (
            self.w_path * path_cost
            + self.w_goal * goal_cost
            + self.w_obs * obstacle_cost
            + self.w_vel * vel_cost
            + self.w_vy * vy_cost
        )  # [B, Nc]

        # ===== B) near-goal yaw align =====
        dist_to_goal_now = torch.linalg.norm(base_xy - goal_xy, dim=-1)  # [B]
        goal_pos_thresh = getattr(self, "goal_pos_threshold", 0.15)
        near_goal_mask = dist_to_goal_now < goal_pos_thresh

        goal_yaw = waypoints_yaw[:, -1]  # [B]
        yaw_err_now = angle_diff(goal_yaw, base_yaw)  # [B]

        # 两种情况进入 turning-only：到达终点附近 or 当前 yaw 偏差超过阈值
        large_yaw_mask = yaw_err_now.abs() > self.turn_yaw_threshold
        turn_only_mask = near_goal_mask | large_yaw_mask

        actions = torch.zeros(B, 3, device=self.device)  # (vx, vy, rz)

        # (B1) turn-only: vx=vy=0, rz=P*yaw_err
        if turn_only_mask.any():
            idx_turn = torch.nonzero(turn_only_mask, as_tuple=False).squeeze(-1)
            # print("idx_turn: ", idx_turn)
            # k_yaw = getattr(self, "k_goal_yaw", 1.0)
            yaw_err = yaw_err_now[idx_turn]
            rz_cmd = torch.sign(yaw_err) * 1  # 只看方向

            actions[idx_turn, 0] = 0.0
            actions[idx_turn, 1] = 0.0
            actions[idx_turn, 2] = rz_cmd

        # (B2) normal follow: argmin cost
        follow_mask = ~turn_only_mask
        if follow_mask.any():
            idx_follow = torch.nonzero(follow_mask, as_tuple=False).squeeze(-1)

            total_cost_follow = total_cost[follow_mask]  # [Nf, Nc]
            best_idx_follow = torch.argmin(total_cost_follow, dim=1)

            actions[idx_follow, :] = candidates[idx_follow, best_idx_follow, :]

        debug_info = {
            "trajs": trajs,
            "candidates": candidates,
            "total_cost": total_cost,
            "path_cost": path_cost,
            "goal_cost": goal_cost,
            "obstacle_cost": obstacle_cost,
            "vel_cost": vel_cost,
            "best_idx_global": torch.argmin(total_cost, dim=1),
            "dist_to_goal_now": dist_to_goal_now,
            "yaw_err_now": yaw_err_now,
            "near_goal_mask": near_goal_mask,
            "turn_only_mask": turn_only_mask,
        }
        return actions, debug_info

    @torch.no_grad()
    def sample_controls(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a batch of (v, steer) command pairs for each env.

        Args:
            robot_state: dict-like with:
                - "base_lin_vel": [N_env, 3] linear velocity in base frame
                - "front_steer":  [N_env, 2] left/right steering angles
            env_ids: 1D tensor of env indices to sample for, shape [B].

        Returns:
            actions: tensor of shape [B, Nv*Ns, 2],
                     where each row contains all (v, steer) pairs
                     for that env.
        """
        device = self.device
        env_ids = env_ids.to(device=device, dtype=torch.long)

        base_lin_vel = robot_state["base_lin_vel"].to(device)  # [N_env, 3]
        front_steer = robot_state["front_steer"].to(device)  # [N_env, 2]

        # ----------------------------------------
        # 1) Gather current v and steer per env
        # ----------------------------------------
        # current longitudinal speed in base frame (vx)
        v_cur = base_lin_vel[env_ids, 0]  # [B]
        # current average front steering angle (left/right)
        steer_cur = front_steer[env_ids].mean(dim=-1)  # [B]

        # ----------------------------------------
        # 2) Compute per-env dynamic windows
        #    v in [v_low, v_high]
        #    steer in [steer_low, steer_high]
        # ----------------------------------------
        dt = self.dt

        v_low = torch.clamp(
            v_cur - self.max_accel * dt,
            min=self.min_speed,
            max=self.max_speed,
        )
        v_high = torch.clamp(
            v_cur + self.max_accel * dt,
            min=self.min_speed,
            max=self.max_speed,
        )

        steer_low = torch.clamp(
            steer_cur - self.max_steer_rate * dt,
            min=self.min_steer,
            max=self.max_steer,
        )
        steer_high = torch.clamp(
            steer_cur + self.max_steer_rate * dt,
            min=self.min_steer,
            max=self.max_steer,
        )

        # ----------------------------------------
        # 3) Build per-env linspace for v and steer
        #    v_samples:    [B, Nv]
        #    steer_samples:[B, Ns]
        # ----------------------------------------
        Nv = self.num_speed_samples
        Ns = self.num_steer_samples

        # shared interpolation parameters in [0,1]
        t_v = torch.linspace(0.0, 1.0, Nv, device=device)  # [Nv]
        t_s = torch.linspace(0.0, 1.0, Ns, device=device)  # [Ns]

        # v_samples[i, :] spans [v_low[i], v_high[i]]
        v_samples = (
            v_low.unsqueeze(1) * (1.0 - t_v) + v_high.unsqueeze(1) * t_v
        )  # [B, Nv]
        # steer_samples[i, :] spans [steer_low[i], steer_high[i]]
        steer_samples = (
            steer_low.unsqueeze(1) * (1.0 - t_s) + steer_high.unsqueeze(1) * t_s
        )  # [B, Ns]

        # ----------------------------------------
        # 4) Meshgrid per env (vectorized)
        #    V: [B, Nv, Ns], S: [B, Nv, Ns]
        #    More efficient meshgrid using broadcasting
        # ----------------------------------------
        # v_samples: [B, Nv] -> [B, Nv, 1] -> [B, Nv, Ns] (broadcast along last dim)
        V = v_samples.unsqueeze(-1).expand(-1, -1, Ns)  # [B, Nv, Ns]
        # steer_samples: [B, Ns] -> [B, 1, Ns] -> [B, Nv, Ns] (broadcast along middle dim)
        S = steer_samples.unsqueeze(1).expand(-1, Nv, -1)  # [B, Nv, Ns]

        # Stack into (v, steer) pairs and flatten candidates per env
        # Use contiguous() for better memory layout before reshape
        actions = torch.stack([V, S], dim=-1)  # [B, Nv, Ns, 2]
        actions = actions.contiguous().reshape(actions.shape[0], -1, 2)  # [B, Nv*Ns, 2]

        return actions

    @torch.no_grad()
    def sample_controls_holonomic(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        全向底盘 (ridgebackFranka) 的控制采样：
        - 在 (vx, vy, yaw_rate) 空间做动态窗口采样
        - vx, vy 来自 base_lin_vel[:, 0:2]
        - yaw_rate 来自 base_ang_vel[:, 2]
        返回:
            actions: [B, Nvx * Nvy * Nw, 3]，每个为 [vx, vy, rz]
        """
        device = self.device
        env_ids = env_ids.to(device=device, dtype=torch.long)

        base_lin_vel = robot_state["base_lin_vel"].to(device)  # [N_env, 3]
        base_ang_vel = robot_state["base_ang_vel"].to(device)  # [N_env, 3]

        # 当前速度
        vx_cur = base_lin_vel[env_ids, 0]  # [B]
        vy_cur = base_lin_vel[env_ids, 1]  # [B]
        rz_cur = base_ang_vel[env_ids, 2]  # [B] yaw rate

        dt = self.dt

        # ====== 1) 动态窗口 for vx ======
        vx_low = torch.clamp(
            vx_cur - self.max_accel_vx * dt,
            min=self.min_vx,
            max=self.max_vx,
        )
        vx_high = torch.clamp(
            vx_cur + self.max_accel_vx * dt,
            min=self.min_vx,
            max=self.max_vx,
        )

        # ====== 2) 动态窗口 for vy ======
        vy_low = torch.clamp(
            vy_cur - self.max_accel_vy * dt,
            min=self.min_vy,
            max=self.max_vy,
        )
        vy_high = torch.clamp(
            vy_cur + self.max_accel_vy * dt,
            min=self.min_vy,
            max=self.max_vy,
        )

        # ====== 3) 动态窗口 for yaw rate (rz) ======
        rz_low = torch.clamp(
            rz_cur - self.max_yaw_accel * dt,
            min=self.min_yaw_rate,
            max=self.max_yaw_rate,
        )
        rz_high = torch.clamp(
            rz_cur + self.max_yaw_accel * dt,
            min=self.min_yaw_rate,
            max=self.max_yaw_rate,
        )

        # 采样数
        Nvx = self.num_vx_samples
        Nvy = self.num_vy_samples
        Nw = self.num_yaw_samples

        t_vx = torch.linspace(0.0, 1.0, Nvx, device=device)  # [Nvx]
        t_vy = torch.linspace(0.0, 1.0, Nvy, device=device)  # [Nvy]
        t_w = torch.linspace(0.0, 1.0, Nw, device=device)  # [Nw]

        # === 插值生成每个维度的候选 ===
        # [B, Nvx]
        vx_samples = vx_low.unsqueeze(1) * (1.0 - t_vx) + vx_high.unsqueeze(1) * t_vx
        # [B, Nvy]
        vy_samples = vy_low.unsqueeze(1) * (1.0 - t_vy) + vy_high.unsqueeze(1) * t_vy
        # [B, Nw]
        rz_samples = rz_low.unsqueeze(1) * (1.0 - t_w) + rz_high.unsqueeze(1) * t_w

        # === 做 3D meshgrid，得到所有 (vx, vy, rz) 组合 ===
        # vx: [B, Nvx, Nvy, Nw]
        VX = vx_samples.unsqueeze(2).unsqueeze(3).expand(-1, Nvx, Nvy, Nw)
        # vy: [B, Nvx, Nvy, Nw]
        VY = vy_samples.unsqueeze(1).unsqueeze(3).expand(-1, Nvx, Nvy, Nw)
        # rz: [B, Nvx, Nvy, Nw]
        RZ = rz_samples.unsqueeze(1).unsqueeze(1).expand(-1, Nvx, Nvy, Nw)

        # 组合为 [vx, vy, rz]
        actions = torch.stack([VX, VY, RZ], dim=-1)  # [B, Nvx, Nvy, Nw, 3]
        actions = actions.contiguous().view(actions.shape[0], -1, 3)  # [B, Ncand, 3]

        return actions

    @torch.no_grad()
    def sample_controls_unicycle(
        self,
        robot_state: dict,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        差分驱动版控制采样：
        - 动态窗口在 (v, omega) 空间采样
        - v 来自 base_lin_vel[:, 0]
        - omega 来自 base_ang_vel[:, 2]
        返回 shape: [B, Nv*Ns, 2]，每个 (v, omega)
        """
        device = self.device
        env_ids = env_ids.to(device=device, dtype=torch.long)
        # print("robot_state: ", robot_state)
        # raise Exception("Stop here")
        base_lin_vel = robot_state["base_lin_vel"].to(device)  # [N_env, 3]
        base_ang_vel = robot_state["base_ang_vel"].to(device)  # [N_env, 3]
        # 当前线速度/角速度
        v_cur = base_lin_vel[env_ids, 0]  # [B]
        w_cur = base_ang_vel[env_ids, 2]  # [B] yaw rate

        dt = self.dt

        # 先把 v_cur/w_cur 夹到合法范围，避免实测偏差导致动态窗口退化
        v_cur_clamped = torch.clamp(v_cur, min=self.min_speed, max=self.max_speed)
        w_cur_clamped = torch.clamp(w_cur, min=self.min_steer, max=self.max_steer)

        # 动态窗口
        v_low = torch.clamp(
            v_cur_clamped - self.max_accel * dt,
            min=self.min_speed,
            max=self.max_speed,
        )
        v_high = torch.clamp(
            v_cur_clamped + self.max_accel * dt,
            min=self.min_speed,
            max=self.max_speed,
        )

        w_low = torch.clamp(
            w_cur_clamped - self.max_steer_rate * dt,
            min=self.min_steer,
            max=self.max_steer,
        )
        w_high = torch.clamp(
            w_cur_clamped + self.max_steer_rate * dt,
            min=self.min_steer,
            max=self.max_steer,
        )

        Nv = self.num_speed_samples
        Nw = self.num_steer_samples

        t_v = torch.linspace(0.0, 1.0, Nv, device=device)  # [Nv]
        t_w = torch.linspace(0.0, 1.0, Nw, device=device)  # [Nw]

        v_samples = (
            v_low.unsqueeze(1) * (1.0 - t_v) + v_high.unsqueeze(1) * t_v
        )  # [B, Nv]
        w_samples = (
            w_low.unsqueeze(1) * (1.0 - t_w) + w_high.unsqueeze(1) * t_w
        )  # [B, Nw]
        # print("min |w_samples|:", w_samples.abs().min())

        # meshgrid
        V = v_samples.unsqueeze(-1).expand(-1, -1, Nw)  # [B, Nv, Nw]
        W = w_samples.unsqueeze(1).expand(-1, Nv, -1)  # [B, Nv, Nw]

        actions = torch.stack([V, W], dim=-1)  # [B, Nv, Nw, 2]
        actions = actions.contiguous().view(actions.shape[0], -1, 2)  # [B, Nv*Nw, 2]

        return actions

    def step(self, target_pos, env_ids):
        if self.step_counter % max(1, self.occupancy_update_interval) == 0:
            self.occupancy = self.generate_occupancy(env_ids)
            self.dilated_occupancy = self.dilate_occupancy(
                self.occupancy, self.dilate_radius, self.map_resolution
            )

        robot_state = self.robot_manager.get_robot_state(noise_flag=False)[0][
            self.robot_name
        ]

        if self.controller_type == "unicycle":
            candidates = self.sample_controls_unicycle(robot_state, env_ids)
        elif self.controller_type == "holonomic":
            candidates = self.sample_controls_holonomic(robot_state, env_ids)
        else:
            candidates = self.sample_controls(robot_state, env_ids)

        # 4) rollout
        if self.controller_type == "unicycle":
            trajs, collided = self.rollout_unicycle(
                robot_state=robot_state,
                env_ids=env_ids,
                controls_per_env=candidates,
                occupancy=self.dilated_occupancy,
                num_steps=self.num_rollout_steps,
            )
        elif self.controller_type == "holonomic":
            trajs, collided = self.rollout_holonomic(
                robot_state=robot_state,
                env_ids=env_ids,
                controls_per_env=candidates,
                occupancy=self.dilated_occupancy,
                num_steps=self.num_rollout_steps,
            )
        else:
            trajs, collided = self.rollout(
                robot_state=robot_state,
                env_ids=env_ids,
                controls_per_env=candidates,
                occupancy=self.dilated_occupancy,
                num_steps=self.num_rollout_steps,
            )

        if self.controller_type == "holonomic":
            actions, dbg = self.evaluate_and_select_actions_holonomic(
                trajs_list=trajs,
                collided_list=collided,
                candidates=candidates,
                target_pos=target_pos,
            )
        else:
            actions, dbg = self.evaluate_and_select_actions(
                trajs_list=trajs,
                collided_list=collided,
                candidates=candidates,
                target_pos=target_pos,
            )

        # trajs = dbg["trajs"]  # List[B], each [Nc,T,3]
        # total_cost = dbg["total_cost"]  # List[B], each [Nc]
        # candidates = dbg["candidates"]  # [B,Nc,2]
        # best_idx = dbg["best_idx_global"]  # [B]
        # self.debug_show_random_20_actions(
        #     occupancy=self.dilated_occupancy,
        #     trajs_list=trajs,
        #     total_cost=total_cost,  # [B, Nc]
        #     candidates=candidates,  # [B, Nc, 2]
        #     best_idx=best_idx,  # [B]
        #     num_show=1,
        # )

        if self.robot_type == "humanoid":
            height_cmd = target_pos[:, -1, 3]  # [B]
            # NaN-fallback: when callers (e.g. Nav GlobalPlanner padding) leave the
            # height column unset, hold the current pelvis z so WBC keeps tracking.
            if torch.isnan(height_cmd).any():
                base_pos_all = self.robot_manager.get_robot_state(noise_flag=False)[0][
                    self.robot_name
                ]["base_pos"]
                cur_height = base_pos_all[env_ids.to(base_pos_all.device), 2].to(
                    height_cmd.device
                )
                height_cmd = torch.where(
                    torch.isnan(height_cmd), cur_height, height_cmd
                )
            # --- append height + 3 zeros for torso_roll, torso_pitch, torso_yaw -> [B,7] ---
            B = actions.shape[0]
            zeros_padding = torch.zeros(
                B, 3, device=actions.device, dtype=actions.dtype
            )
            # print("actions: ", actions)
            actions = torch.cat(
                [
                    actions,
                    height_cmd.unsqueeze(-1).to(
                        device=actions.device, dtype=actions.dtype
                    ),
                    zeros_padding,
                ],
                dim=-1,
            )
        if self.debug:
            try:
                v_np = actions.detach().cpu().numpy()
                print(
                    f"[Dwb] step={self.step_counter} env_ids={env_ids.tolist()} "
                    f"actions(vx,vy,rz)={v_np.tolist()}"
                )
            except Exception:
                pass
        if self.robot_type == "quadruped":
            height_cmd = target_pos[:, -1, 3]  # [B]
            if torch.isnan(height_cmd).any():
                base_pos_all = self.robot_manager.get_robot_state(noise_flag=False)[0][
                    self.robot_name
                ]["base_pos"]
                cur_height = base_pos_all[env_ids.to(base_pos_all.device), 2].to(
                    height_cmd.device
                )
                height_cmd = torch.where(
                    torch.isnan(height_cmd), cur_height, height_cmd
                )
            B = actions.shape[0]
            actions = torch.cat(
                [
                    actions,
                    height_cmd.unsqueeze(-1).to(
                        device=actions.device, dtype=actions.dtype
                    ),
                ],
                dim=-1,
            )
        self.step_counter += 1

        return actions

    def update_obstacles(
        self, obstacle_avoidance_path_list, obstacle_ignore_path_list, env_ids
    ):
        pass

    def reset_idx(self, env_ids):
        pass

    def debug_show_random_20_actions(
        self,
        occupancy,  # [B,H,W]
        trajs_list,  # List[B], each [Nc,T,3]
        total_cost,  # [B,Nc]
        candidates,  # [B,Nc,2]
        best_idx,  # [B]
        num_show=20,
    ):
        """
        调试 env0：
        - 随机抽 num_show 个 action
        - 弹窗显示 rollout 轨迹、最优轨迹、costmap
        """
        import numpy as np

        # ---- 取 env=0 ----
        trajs0 = trajs_list[0]  # [Nc,T,3]
        cost0 = total_cost[0]  # [Nc]
        cand0 = candidates[0]  # [Nc,2]
        best_k = best_idx[0].item()
        occ0 = occupancy[0]  # [H,W]
        best_cost = cost0[best_k].item()

        Nc = trajs0.shape[0]
        K = min(num_show, Nc)
        sample_ids = np.random.choice(Nc, K, replace=False)

        print(f"[DWB DEBUG] env0: show {K} actions {sample_ids}, best={best_k}")

        # ---- 准备可视化地图 ----
        occ_np = (1 - occ0.detach().cpu().numpy()) * 255
        occ_rgb = np.stack([occ_np] * 3, axis=-1).astype(np.uint8)

        # ---- map meta (local) ----
        ox, oy, res = self.map_meta_local_all[0].tolist()

        for k in sample_ids:
            traj = trajs0[k].detach().cpu().numpy()  # [T,3]
            traj_best = trajs0[best_k].detach().cpu().numpy()

            xs, ys = traj[:, 0], traj[:, 1]
            xs_b, ys_b = traj_best[:, 0], traj_best[:, 1]

            # world -> pixel
            js = (xs - ox) / res
            is_ = (ys - oy) / res
            jsb = (xs_b - ox) / res
            isb = (ys_b - oy) / res

            # ---- 画图 ----
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(occ_rgb, origin="lower")

            # 当前 candidate（青色）
            ax.plot(js, is_, "c-", linewidth=2)
            ax.scatter(js[0], is_[0], c="g", s=40)  # 起点
            ax.scatter(js[-1], is_[-1], c="r", s=40)  # 终点

            # 最优轨迹（黄色）
            ax.plot(jsb, isb, "y--", linewidth=1.5, alpha=0.8)

            # 文本
            ax.set_title(
                f"env0 | action {k} | cost={cost0[k]:.3f} | best_cost={best_cost:.3f} | v={cand0[k, 0]:.2f} steer={cand0[k, 1]:.2f}"
            )
            plt.savefig("/home/magic/shuyang/MagicSim/dwb_debug_action.png")
            # plt.tight_layout()
            # plt.show()
