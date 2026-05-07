# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides methods to create different terrains using the ``trimesh`` library.

In contrast to the height-field representation, the trimesh representation does not
create arbitrarily small triangles. Instead, the terrain is represented as a single
tri-mesh primitive. Thus, this representation is more computationally and memory
efficient than the height-field representation, but it is not as flexible.
"""

from .mesh_terrains_cfg import (
    MeshBoxTerrainCfg,  # noqa: F401, F403
    MeshFloatingRingTerrainCfg,  # noqa: F401, F403
    MeshGapTerrainCfg,  # noqa: F401, F403
    MeshInvertedPyramidStairsTerrainCfg,  # noqa: F401, F403
    MeshPitTerrainCfg,  # noqa: F401, F403
    MeshPlaneTerrainCfg,  # noqa: F401, F403
    MeshPyramidStairsTerrainCfg,  # noqa: F401, F403
    MeshRailsTerrainCfg,  # noqa: F401, F403
    MeshRandomGridTerrainCfg,  # noqa: F401, F403
    MeshRepeatedBoxesTerrainCfg,  # noqa: F401, F403
    MeshRepeatedCylindersTerrainCfg,  # noqa: F401, F403
    MeshRepeatedPyramidsTerrainCfg,  # noqa: F401, F403
    MeshStarTerrainCfg,  # noqa: F401, F403
)
