# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from isaaclab.utils import configclass
from dataclasses import MISSING
from . import terrains as terrain_gen


@configclass
class MagicTerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    size: int = MISSING
    num_rows: int = MISSING
    num_cols: int = MISSING
    sub_terrains: dict[str, terrain_gen.SubTerrainBaseCfg] = None
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = 0.75
    use_cache = False
    sub_terrains = {
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.0,
        #     noise_range=(0.02, 0.10),
        #     noise_step=0.02,
        #     border_width=0.25,
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.0, slope_range=[0.3, 0.5]
        # ),
        "hf_pyramid_stair": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.08, 0.23),
            step_width=0.35,
            platform_width=3.0,
            border_width=0.1,
        ),
        # "hf_pyramid_stair_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
        #     proportion=0.0,
        #     step_height_range=(0.05, 0.15),
        #     step_width=0.5,
        #     platform_width=3.0,
        #     border_width=0.1,
        # ),
        # "discreate": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.0,
        #     obstacle_width_range=[0.4, 0.7],
        #     obstacle_height_range=[0.1, 0.2],
        #     num_obstacles=250,
        # ),
        # "wave": terrain_gen.HfWaveTerrainCfg(
        #     proportion=0.5, amplitude_range=[0.3, 0.5], num_waves=2
        # ),
        # "stone": terrain_gen.HfSteppingStonesTerrainCfg(
        #     proportion=0.5,
        #     stone_height_max=0.3,
        #     stone_width_range=[0.5, 0.7],
        #     stone_distance_range=[0.2, 0.4],
        # ),
        # "magic_stair": terrain_gen.StairsTerrainCfg(
        #     proportion=0.5,
        #     step_height_range=[0.1, 0.15],
        #     step_width_range=[0.4, 0.3],
        #     platform_width=0.8,
        #     border_width=0.1,
        # ),
        "magic_parkour": terrain_gen.ParkourTerrainCfg(
            proportion=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.0],
            num_goals=12,
        ),
        "magic_hurdle": terrain_gen.HurdleTerrainCfg(
            proportion=0.0,
            hurdle_range=[0.1, 0.3],
            hurdle_height_range=[0.08, 0.18],
            flat_size=0.8,
        ),
        "magic_bridge": terrain_gen.BridgeTerrainCfg(
            proportion=0.5,
            bridge_width_range=[0.5, 0.4],
            bridge_height=0.7,
            platform_width=1.5,
        ),
        # "magic_platform": terrain_gen.PlatformTerrainCfg(
        #     proportion=0.5,
        #     height_range=[0.05, 0.15],
        #     flat_size=1.0,
        #     platform_width=1.0,
        #     border_width=0.1,
        # ),
        # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0),
        # "star": terrain_gen.MeshStarTerrainCfg(proportion=1.0, num_bars=3, bar_height_range=(0.0, 0.3), bar_width_range=(0.1, 0.2)),
        # "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(proportion=0.4, ring_height_range=(0.0, 0.8), ring_width_range=(1.2, 3.6), ring_thickness=0.2),
        # "box": terrain_gen.MeshBoxTerrainCfg(proportion=0.3, box_height_range=(0, 1.5), double_box=True),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.0,
        #     step_height_range=(0.06, 0.16),
        #     step_width=0.25,
        #     platform_width=2.0,
        # ),
        # "invert_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.0,
        #     step_height_range=(0.06, 0.12),
        #     step_width=0.25,
        #     platform_width=2.0,
        # ),
        # "hf_random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.6,
        #     noise_range=(-0.06, 0.06),
        #     noise_step=0.005
        #     ),
        # "hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #         proportion=0.4,
        #         platform_width=2,
        #         obstacle_width_range=(1, 2),
        #         obstacle_height_range=(0.1, 0.15),
        #         num_obstacles=2000,
        #           ),
    }


# Magic_terrain = MagicTerrainGeneratorCfg(
#     size=(6.0, 4.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum=True,
#     sub_terrains={
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
#         ),
#         "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.0, slope_range=[0.3, 0.5]
#         ),
#         "hf_pyramid_stair": terrain_gen.HfPyramidStairsTerrainCfg(
#             proportion=0.0,
#             step_height_range=(0.08, 0.23),
#             step_width=0.35,
#             platform_width=3.0,
#             border_width=0.1,
#         ),
#         "hf_pyramid_stair_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
#             proportion=0.0,
#             step_height_range=(0.05, 0.15),
#             step_width=0.5,
#             platform_width=3.0,
#             border_width=0.1,
#         ),
#         "discreate": terrain_gen.HfDiscreteObstaclesTerrainCfg(
#             proportion=0.0,
#             obstacle_width_range=[0.4, 0.7],
#             obstacle_height_range=[0.1, 0.2],
#             num_obstacles=250,
#         ),
#         "wave": terrain_gen.HfWaveTerrainCfg(
#             proportion=0.0, amplitude_range=[0.3, 0.5], num_waves=2
#         ),
#         "stone": terrain_gen.HfSteppingStonesTerrainCfg(
#             proportion=0.0,
#             stone_height_max=0.3,
#             stone_width_range=[0.5, 0.7],
#             stone_distance_range=[0.2, 0.4],
#         ),
#         "magic_stair": terrain_gen.StairsTerrainCfg(
#             proportion=0.5,
#             step_height_range=[0.1, 0.15],
#             step_width_range=[0.4, 0.3],
#             platform_width=0.8,
#             border_width=0.1,
#         ),
#         "magic_parkour": terrain_gen.ParkourTerrainCfg(
#             proportion=0.0,
#             x_range=[0.5, 1.0],
#             y_range=[0.3, 0.4],
#             stone_len_range=[0.8, 1.0],
#             stone_width_range=[0.6, 0.8],
#             incline_height=0.1,
#             pit_depth=[0.5, 1.0],
#             num_goals=12,
#         ),
#         #  "magic_hurdle":terrain_gen.HurdleTerrainCfg(
#         #     proportion=0.0,
#         #     hurdle_range=[0.1, 0.3],
#         #     hurdle_height_range=[0.08, 0.18],
#         #     flat_size = 0.8
#         # ),（remains problem）
#         "magic_bridge": terrain_gen.BridgeTerrainCfg(
#             proportion=0.0,
#             bridge_width_range=[0.5, 0.4],
#             bridge_height=0.7,
#             platform_width=1.5,
#         ),
#         "magic_platform": terrain_gen.PlatformTerrainCfg(
#             proportion=0.0,
#             height_range=[0.05, 0.15],
#             flat_size=1.0,
#             platform_width=1.0,
#             border_width=0.1,
#         ),
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0),
#         # "random_grid": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.2, grid_width=0.06, grid_height_range=(0, 0.03)),
#         # "star": terrain_gen.MeshStarTerrainCfg(proportion=1.0, num_bars=3, bar_height_range=(0.0, 0.3), bar_width_range=(0.1, 0.2)),
#         # "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(proportion=0.4, ring_height_range=(0.0, 0.8), ring_width_range=(1.2, 3.6), ring_thickness=0.2),
#         # "box": terrain_gen.MeshBoxTerrainCfg(proportion=0.3, box_height_range=(0, 1.5), double_box=True),
#         "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.0,
#             step_height_range=(0.06, 0.16),
#             step_width=0.25,
#             platform_width=2.0,
#         ),
#         "invert_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.0,
#             step_height_range=(0.06, 0.12),
#             step_width=0.25,
#             platform_width=2.0,
#         ),
#         # "hf_random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
#         #     proportion=0.6,
#         #     noise_range=(-0.06, 0.06),
#         #     noise_step=0.005
#         #     ),
#         # "hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
#         #         proportion=0.4,
#         #         platform_width=2,
#         #         obstacle_width_range=(1, 2),
#         #         obstacle_height_range=(0.1, 0.15),
#         #         num_obstacles=2000,
#         #           ),
#     },
# )
