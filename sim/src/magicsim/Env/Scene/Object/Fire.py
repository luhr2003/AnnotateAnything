# Object/Fire.py

from omegaconf import DictConfig
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.stage import get_current_stage
import omni.kit.commands
import carb.settings


class Fire:
    def __init__(
        self,
        prim_path: str,
        config: DictConfig,
    ):
        """
        Create a fire simulation using NVIDIA Flow preset.

        Args:
            prim_path: The unique prim path for this fire instance
            config: Configuration for fire parameters
        """
        self.prim_path = prim_path
        self.config = config
        self.stage = get_current_stage()

        # Get fire parameters from config with defaults
        # Transform parameters
        self.position = config.get("pos", [0, 0, 0])
        self.scale = config.get("scale", 0.1)

        # Emitter geometry
        self.radius = config.get("radius", 10.0)

        # Emitter physics parameters (CRITICAL for realistic fire!)
        self.velocity = config.get("velocity", 150.0)  # Upward speed
        self.temperature = config.get("temperature", 3.0)  # Heat value
        self.density_emit = config.get("density_emit", 400.0)  # Emission intensity

        # Simulation physics parameters (makes fire look alive!)
        self.buoyancy = config.get("buoyancy", 2.0)  # Rising force
        self.vorticity = config.get("vorticity", 40.0)  # Turbulent swirling
        self.dissipation = config.get("dissipation", 0.15)  # Cooling rate
        self.divergence = config.get("divergence", -150.0)  # Expansion
        self.cell_size = config.get("cell_size", 0.5)  # Simulation resolution

        # Rendering parameters
        self.temperature_scale = config.get("temperature_scale", 1.5)
        self.density_scale = config.get("density_scale", 2.0)

        # Render layer (CRITICAL: each fire must have unique layer!)
        self.layer = config.get("layer", 0)

        # Color
        colormap = config.get("colormap", "fire")

        print(f"\n{'=' * 60}")
        print(f"Creating fire at: {self.prim_path}")
        print(f"{'=' * 60}")
        print("Transform:")
        print(f"  Position: {self.position}")
        print(f"  Scale: {self.scale}")
        print("\nEmitter Geometry:")
        print(f"  Radius: {self.radius}")
        print("\nEmitter Physics:")
        print(f"  Velocity: {self.velocity}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Density Emit: {self.density_emit}")
        print("\nSimulation Physics:")
        print(f"  Buoyancy: {self.buoyancy}")
        print(f"  Vorticity: {self.vorticity}")
        print(f"  Dissipation: {self.dissipation}")
        print(f"  Divergence: {self.divergence}")
        print(f"  Cell Size: {self.cell_size}")
        print("\nRendering:")
        print(f"  Temperature Scale: {self.temperature_scale}")
        print(f"  Density Scale: {self.density_scale}")
        print(f"  Layer: {self.layer}")
        print(f"  Colormap: {colormap}")
        print(f"{'=' * 60}")

        # Define color presets
        self.color_presets = {
            "fire": [
                (0.0154, 0.0177, 0.0154, 0.004902),
                (0.03575, 0.03575, 0.03575, 0.504902),
                (0.03575, 0.03575, 0.03575, 0.504902),
                (1.0, 0.1594, 0.0134, 0.8),
                (13.53, 2.99, 0.12599, 0.8),
                (78, 39, 6.1, 0.7),
            ],
            "blue": [
                (0.01, 0.01, 0.02, 0.004902),
                (0.02, 0.02, 0.05, 0.504902),
                (0.1, 0.3, 1.0, 0.504902),
                (0.5, 2.0, 8.0, 0.8),
                (2.0, 10.0, 25.0, 0.8),
                (10.0, 40.0, 80.0, 0.7),
            ],
            "green": [
                (0.01, 0.02, 0.01, 0.004902),
                (0.02, 0.05, 0.02, 0.504902),
                (0.1, 1.0, 0.1, 0.504902),
                (0.5, 8.0, 0.5, 0.8),
                (2.0, 25.0, 2.0, 0.8),
                (10.0, 80.0, 10.0, 0.7),
            ],
            "purple": [
                (0.02, 0.01, 0.02, 0.004902),
                (0.05, 0.02, 0.05, 0.504902),
                (0.8, 0.1, 1.0, 0.504902),
                (5.0, 0.5, 8.0, 0.8),
                (15.0, 2.0, 25.0, 0.8),
                (60.0, 10.0, 80.0, 0.7),
            ],
            "white": [
                (0.01, 0.01, 0.01, 0.004902),
                (0.1, 0.1, 0.1, 0.504902),
                (0.5, 0.5, 0.5, 0.504902),
                (5.0, 5.0, 5.0, 0.8),
                (20.0, 20.0, 20.0, 0.8),
                (80.0, 80.0, 80.0, 0.7),
            ],
        }

        # Get colormap
        if isinstance(colormap, str):
            self.rgba_points = self.color_presets.get(
                colormap.lower(), self.color_presets["fire"]
            )
        else:
            self.rgba_points = colormap

        # Enable Flow render settings
        self._enable_flow_rendering()

        # Create the fire
        self._create_fire()

    def _enable_flow_rendering(self):
        """Enable Flow extension and rendering in RTX settings."""
        import time

        # CRITICAL: Enable Flow extension first!
        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            if not ext_manager.is_extension_enabled("omni.flowusd"):
                ext_manager.set_extension_enabled_immediate("omni.flowusd", True)
                print("✓ Flow extension enabled")
                # Give extension time to fully initialize
                time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not enable Flow extension: {e}")
            print("  Flow may already be enabled or app not fully initialized")

        # Enable Flow rendering in RTX settings
        settings = carb.settings.get_settings()
        settings.set("/rtx/flow/enabled", True)
        settings.set("/rtx/flow/pathTracingEnabled", True)
        settings.set("/rtx/flow/rayTracedReflectionsEnabled", True)
        settings.set("/rtx/flow/rayTracedTranslucencyEnabled", True)

    def _create_fire(self):
        """Create fire using Flow preset."""
        # Create root xform for positioning and scaling
        root_xform = UsdGeom.Xform.Define(self.stage, self.prim_path)
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*self.position))
        root_xform.AddScaleOp().Set(Gf.Vec3f(self.scale, self.scale, self.scale))
        print(f"✓ Created fire root transform at {self.prim_path}")

        # Create Fire preset using Flow command
        success, created_prims = omni.kit.commands.execute(
            "FlowCreatePresetsCommand",
            preset_name="Fire",
            paths=[self.prim_path],
            create_copy=True,
            layer=self.layer,
            url="",
        )

        if not success:
            print(f"❌ Failed to create Fire preset at {self.prim_path}")
            return
        else:
            print(f"✓ Fire preset created successfully at {self.prim_path}")
            print(f"  Created prims: {created_prims}")

        # Adjust fire parameters
        root_layer = self.stage.GetRootLayer()

        with Usd.EditContext(self.stage, root_layer):
            fire_prim = self.stage.GetPrimAtPath(self.prim_path)

            if fire_prim.IsValid():
                print("\nConfiguring Flow Emitter...")
                # Configure emitter parameters
                for child in fire_prim.GetAllChildren():
                    if "FlowEmitter" in child.GetTypeName():
                        # Geometry
                        radius_attr = child.GetAttribute("radius")
                        if radius_attr:
                            radius_attr.Set(float(self.radius))
                            print(f"  ✓ Radius: {self.radius}")

                        # Physics - CRITICAL for realistic fire!
                        # Velocity is a 3D vector (upward = [0, 0, z])
                        velocity_attr = child.GetAttribute("velocity")
                        if velocity_attr:
                            velocity_vec = Gf.Vec3f(0.0, 0.0, float(self.velocity))
                            velocity_attr.Set(velocity_vec)
                            print(f"  ✓ Velocity: {velocity_vec}")

                        temperature_attr = child.GetAttribute("temperature")
                        if temperature_attr:
                            temperature_attr.Set(float(self.temperature))
                            print(f"  ✓ Temperature: {self.temperature}")

                        density_emit_attr = child.GetAttribute("densityEmit")
                        if density_emit_attr:
                            density_emit_attr.Set(float(self.density_emit))
                            print(f"  ✓ Density Emit: {self.density_emit}")

                print("\nConfiguring Flow Simulation...")
                # Configure simulation parameters
                simulate_prim = self.stage.GetPrimAtPath(
                    f"{self.prim_path}/flowSimulate"
                )
                if simulate_prim.IsValid():
                    # Resolution
                    cell_size_attr = simulate_prim.GetAttribute("densityCellSize")
                    if cell_size_attr:
                        cell_size_attr.Set(float(self.cell_size))
                        print(f"  ✓ Cell Size: {self.cell_size}")

                    # Physics - makes fire look alive!
                    buoyancy_attr = simulate_prim.GetAttribute("buoyancy")
                    if buoyancy_attr:
                        buoyancy_attr.Set(float(self.buoyancy))
                        print(f"  ✓ Buoyancy: {self.buoyancy}")

                    vorticity_attr = simulate_prim.GetAttribute("vorticityConfinement")
                    if vorticity_attr:
                        vorticity_attr.Set(float(self.vorticity))
                        print(f"  ✓ Vorticity: {self.vorticity}")

                    dissipation_attr = simulate_prim.GetAttribute("dissipationRate")
                    if dissipation_attr:
                        dissipation_attr.Set(float(self.dissipation))
                        print(f"  ✓ Dissipation: {self.dissipation}")

                    divergence_attr = simulate_prim.GetAttribute("divergence")
                    if divergence_attr:
                        divergence_attr.Set(float(self.divergence))
                        print(f"  ✓ Divergence: {self.divergence}")

                print("\nConfiguring Flow Rendering...")
                # Configure rendering parameters
                offscreen_prim = self.stage.GetPrimAtPath(
                    f"{self.prim_path}/flowOffscreen"
                )
                if offscreen_prim.IsValid():
                    temp_scale_attr = offscreen_prim.GetAttribute("temperatureScale")
                    if temp_scale_attr:
                        temp_scale_attr.Set(float(self.temperature_scale))
                        print(f"  ✓ Temperature Scale: {self.temperature_scale}")

                    density_scale_attr = offscreen_prim.GetAttribute("densityScale")
                    if density_scale_attr:
                        density_scale_attr.Set(float(self.density_scale))
                        print(f"  ✓ Density Scale: {self.density_scale}")

                # Update colormap
                print("\nConfiguring Colormap...")
                colormap_paths = [
                    f"{self.prim_path}/flowOffscreen/colormap",
                    f"{self.prim_path}/flowOffscreen/rayMarch/colormap",
                ]

                for cmap_path in colormap_paths:
                    colormap_prim = self.stage.GetPrimAtPath(cmap_path)
                    if colormap_prim.IsValid():
                        rgba_attr = colormap_prim.GetAttribute("rgbaPoints")
                        if rgba_attr:
                            rgba_attr.Clear()
                            rgba_attr.Set(self.rgba_points)
                            print("  ✓ Colormap applied")
                            break

        print(f"\n{'=' * 60}")
        print("✓ Fire created successfully - ready to simulate!")
        print("  Note: Fire needs 100+ simulation steps to become visible")
        print(f"{'=' * 60}\n")

    def reset(self):
        """Reset fire parameters (optional, can implement randomization here)."""
        # If you want to randomize fire position/color on reset, implement here
        pass
