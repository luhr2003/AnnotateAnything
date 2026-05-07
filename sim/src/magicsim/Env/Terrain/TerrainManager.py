from math import sqrt
from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from isaaclab.terrains import TerrainImporterCfg  # noqa F401
import isaaclab.sim as sim_utils
from magicsim.Env.Terrain.TerrainGeneratorCfg import MagicTerrainGeneratorCfg
from . import terrains as terrain_gen


class TerrainManager:
    def __init__(self, num_envs, env_spacing, config, device):
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.config = config
        self.device = device
        if self.config is None:
            self.type = "plane"

    def initialize(self, sim: IsaacRLEnv):
        """
        Please Import Terrain Here!!!
        This function will be called in _setup_scene function of SyncBaseEnv
        This function will be called before simulation context create
        Remember to write terrain to the sim.scene
        Please use the same num_envs and env_spacing as the parameter because it will be overwrite later in the cloner
        """
        if self.config is None or self.config.type == "plane":
            self.terrain_cfg = TerrainImporterCfg(
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
                prim_path="/World/ground",
                max_init_terrain_level=None,
                terrain_type="plane" if self.config is None else self.config.type,
                terrain_generator=None,
                debug_vis=False,
            )
        else:
            # self.terrain_generator_cfg = ROUGH_TERRAINS_CFG
            self.terrain_generator_cfg = MagicTerrainGeneratorCfg()
            self.terrain_generator_cfg.size = (self.env_spacing, self.env_spacing)
            self.terrain_generator_cfg.num_rows = int(sqrt(self.num_envs))
            self.terrain_generator_cfg.num_cols = int(sqrt(self.num_envs))
            print("Terrain Generator Config:", self.terrain_generator_cfg)

            self.terrain_cfg = TerrainImporterCfg(
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=self.terrain_generator_cfg,
                max_init_terrain_level=5,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                debug_vis=True,
            )

        self.terrain = terrain_gen.TerrainImporter(self.terrain_cfg)
        sim.scene._terrain = self.terrain
