# Object/Flow.py

from omegaconf import DictConfig
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.stage import get_current_stage
import omni.kit.commands
import carb.settings
import time


class Flow:
    def __init__(
        self,
        prim_path: str,
        config: DictConfig,
    ):
        """
        Create a flow simulation (Smoke, Steam, Dust) using NVIDIA Flow preset.

        Args:
            prim_path: The unique prim path for this flow instance
            config: Configuration for flow parameters
        """
        self.prim_path = prim_path
        self.config = config
        self.stage = get_current_stage()

        # Get flow type
        self.flow_type = config.get("type", "smoke").lower()  # smoke, steam, dust

        # Transform parameters
        self.position = config.get("pos", [0, 0, 0])
        self.scale = config.get("scale", 1.0)

        # Render layer
        self.layer = config.get("layer", 0)

        # Get intensity
        self.intensity = config.get("intensity", 1.0)

        # Get flow type configuration
        if self.flow_type == "smoke":
            self._setup_smoke_defaults()
        elif self.flow_type == "steam":
            self._setup_steam_defaults()
        elif self.flow_type == "dust":
            self._setup_dust_defaults()
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")

        # Override defaults with config values
        self._apply_config_overrides()

        print(f"\n{'=' * 60}")
        print(f"Creating {self.flow_type} at: {self.prim_path}")
        print(f"{'=' * 60}")
        self._print_config()

        # Enable Flow rendering
        self._enable_flow_rendering()

        # Create the flow effect
        self._create_flow()

    def _setup_smoke_defaults(self):
        """Setup default parameters for dark smoke."""
        # Emitter parameters
        self.smoke_emit = 10.0
        self.couple_rate_smoke = 2.0
        self.fuel_emit = 4.0
        self.couple_rate_fuel = 2.0
        self.temperature_emit = 10.0
        self.couple_rate_temperature = 2.0
        self.burn_emit = 0.0
        self.couple_rate_burn = 0.0
        self.velocity = Gf.Vec3f(0, 0, 400)
        self.couple_rate_velocity = 100.0
        self.divergence_emit = 0.0
        self.couple_rate_divergence = 0.0
        self.emitter_radius = 5.0

        # Simulation parameters
        self.cell_size = 1.0
        self.default_layer = 4

        # Advection parameters
        self.buoyancy_per_smoke = 0.0
        self.buoyancy_per_temp = 10.0
        self.cooling_rate = 1.5
        self.gravity = Gf.Vec3f(0, 0, -100)

        # Combustion parameters
        self.combustion_enabled = True
        self.burn_per_temp = 4.0
        self.ignition_temp = 0.05
        self.temp_per_burn = 5.0
        self.smoke_per_burn = 3.0
        self.fuel_per_burn = 0.25
        self.divergence_per_burn = 4.0

        # Channel fade/damping
        self.smoke_damping = 0.3
        self.smoke_fade = 0.4
        self.velocity_damping = 0.01
        self.velocity_fade = 1.0

        # Vorticity
        self.vorticity_enabled = True
        self.vorticity_force = 2.8

        # Rendering
        self.attenuation = 0.5
        self.raymarch_attenuation = 0.5

        # Color
        color_brightness = self.config.get("color_brightness", 0.1)
        self.rgba_points = [
            (0.0154, 0.0177, 0.0154, 0.004902),
            (color_brightness, color_brightness, color_brightness, 0.504902),
            (color_brightness, color_brightness, color_brightness, 0.504902),
            (color_brightness, color_brightness, color_brightness, 0.8),
            (color_brightness, color_brightness, color_brightness, 0.8),
            (color_brightness, color_brightness, color_brightness, 0.7),
        ]

    def _setup_steam_defaults(self):
        """Setup default parameters for steam."""
        # Emitter parameters - KEY: No direct smoke, uses fuel combustion
        self.smoke_emit = 0.0
        self.couple_rate_smoke = 0.0
        self.fuel_emit = 0.8
        self.couple_rate_fuel = 2.0
        self.temperature_emit = 0.5
        self.couple_rate_temperature = 2.0
        self.burn_emit = 0.0
        self.couple_rate_burn = 0.0
        self.velocity = Gf.Vec3f(0, 0, 400)
        self.couple_rate_velocity = 2.0
        self.divergence_emit = 0.0
        self.couple_rate_divergence = 0.0
        self.emitter_radius = 10.0

        # Simulation parameters
        self.cell_size = 0.5
        self.default_layer = 2

        # Advection parameters - Low buoyancy for gentle rise
        self.buoyancy_per_smoke = 0.0
        self.buoyancy_per_temp = 2.0
        self.cooling_rate = 1.5
        self.gravity = Gf.Vec3f(0, 0, -100)

        # Combustion parameters
        self.combustion_enabled = True
        self.burn_per_temp = 4.0
        self.ignition_temp = 0.05
        self.temp_per_burn = 5.0
        self.smoke_per_burn = 3.0
        self.fuel_per_burn = 0.25
        self.divergence_per_burn = 0.0

        # Channel fade/damping - High fade for quick dissipation
        self.smoke_damping = 0.3
        self.smoke_fade = 0.65
        self.velocity_damping = 0.01
        self.velocity_fade = 1.0

        # Vorticity - Low for gentle movement
        self.vorticity_enabled = True
        self.vorticity_force = 0.6

        # Rendering - Low attenuation for wispy appearance
        self.attenuation = 0.045
        self.raymarch_attenuation = 0.05

        # Color - White/light gray with high opacity
        self.rgba_points = [
            (0.9, 0.9, 0.9, 0.004902),
            (0.9, 0.9, 0.9, 0.904902),
            (0.9, 0.9, 0.9, 0.904902),
            (0.9, 0.9, 0.9, 0.904902),
            (0.9, 0.9, 0.9, 0.904902),
            (0.9, 0.9, 0.9, 0.904902),
        ]

    def _setup_dust_defaults(self):
        """Setup default parameters for dust."""
        # Emitter parameters - KEY: Low smoke with HIGH couple rate, no heat
        self.smoke_emit = 0.5
        self.couple_rate_smoke = 10.0
        self.fuel_emit = 0.0
        self.couple_rate_fuel = 0.0
        self.temperature_emit = 0.0
        self.couple_rate_temperature = 0.0
        self.burn_emit = 0.0
        self.couple_rate_burn = 0.0
        # KEY: Velocity on Y-axis for dust
        self.velocity = Gf.Vec3f(0, 400, 0)
        self.couple_rate_velocity = 2.0
        self.divergence_emit = 0.0
        self.couple_rate_divergence = 0.0
        self.emitter_radius = 40.0

        # Simulation parameters
        self.cell_size = 2.0
        self.default_layer = 1

        # Advection parameters - KEY: NEGATIVE buoyancy makes dust fall
        self.buoyancy_per_smoke = -0.5
        self.buoyancy_per_temp = 10.0
        self.cooling_rate = 1.5
        self.gravity = Gf.Vec3f(0, 0, -100)

        # Combustion parameters (enabled but won't burn without fuel)
        self.combustion_enabled = True
        self.burn_per_temp = 4.0
        self.ignition_temp = 0.05
        self.temp_per_burn = 5.0
        self.smoke_per_burn = 3.0
        self.fuel_per_burn = 0.25
        self.divergence_per_burn = 4.0

        # Channel fade/damping
        self.smoke_damping = 0.3
        self.smoke_fade = 0.15
        self.velocity_damping = 0.01
        self.velocity_fade = 1.0

        # Vorticity
        self.vorticity_enabled = True
        self.vorticity_force = 1.4

        # Rendering
        self.attenuation = 0.5
        self.raymarch_attenuation = 0.5

        # Color - Earth tones
        dust_type = self.config.get("dust_type", "tan")
        color_presets = {
            "tan": (0.64, 0.54, 0.32),
            "brown": (0.45, 0.35, 0.25),
            "gray": (0.4, 0.4, 0.4),
            "dark": (0.2, 0.2, 0.2),
        }
        base_color = color_presets.get(dust_type.lower(), color_presets["tan"])
        self.rgba_points = [
            (*base_color, 0.004902),
            (*base_color, 0.504902),
            (*base_color, 0.504902),
            (*base_color, 0.8),
            (*base_color, 0.8),
            (*base_color, 0.7),
        ]

    def _apply_config_overrides(self):
        """Apply config overrides to default parameters."""
        # Allow config to override any parameter
        self.smoke_emit = (
            self.config.get("smoke_emit", self.smoke_emit) * self.intensity
        )
        self.fuel_emit = self.config.get("fuel_emit", self.fuel_emit) * self.intensity
        self.temperature_emit = (
            self.config.get("temperature_emit", self.temperature_emit) * self.intensity
        )
        self.emitter_radius = self.config.get("radius", self.emitter_radius)
        self.cell_size = self.config.get("cell_size", self.cell_size)
        self.buoyancy_per_smoke = self.config.get(
            "buoyancy_per_smoke", self.buoyancy_per_smoke
        )
        self.buoyancy_per_temp = self.config.get(
            "buoyancy_per_temp", self.buoyancy_per_temp
        )
        self.vorticity_force = self.config.get("vorticity", self.vorticity_force)
        self.smoke_fade = self.config.get("fade", self.smoke_fade)

        # Visibility parameters
        self.attenuation = self.config.get("attenuation", self.attenuation)
        self.raymarch_attenuation = self.config.get(
            "raymarch_attenuation", self.raymarch_attenuation
        )

        # Use config layer or default for type
        if self.layer == 0:
            self.layer = self.default_layer

    def _print_config(self):
        """Print configuration summary."""
        print(f"Type: {self.flow_type.upper()}")
        print(f"Position: {self.position}")
        print(f"Scale: {self.scale}")
        print(f"Layer: {self.layer}")
        print(f"Intensity: {self.intensity}")
        print("\nEmitter:")
        print(
            f"  Smoke: {self.smoke_emit}, Fuel: {self.fuel_emit}, Temp: {self.temperature_emit}"
        )
        print(f"  Radius: {self.emitter_radius}")
        print("\nSimulation:")
        print(f"  Cell Size: {self.cell_size}")
        print(
            f"  Buoyancy (smoke/temp): {self.buoyancy_per_smoke}/{self.buoyancy_per_temp}"
        )
        print(f"  Vorticity: {self.vorticity_force}")
        print("\nVisibility:")
        print(
            f"  Attenuation: {self.attenuation} (shadow), {self.raymarch_attenuation} (raymarch)"
        )
        print(f"{'=' * 60}")

    def _enable_flow_rendering(self):
        """Enable Flow extension and rendering in RTX settings."""
        import time

        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            if not ext_manager.is_extension_enabled("omni.flowusd"):
                ext_manager.set_extension_enabled_immediate("omni.flowusd", True)
                print("✓ Flow extension enabled")
                time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not enable Flow extension: {e}")

        settings = carb.settings.get_settings()
        settings.set("/rtx/flow/enabled", True)
        settings.set("/rtx/flow/pathTracingEnabled", True)
        settings.set("/rtx/flow/rayTracedReflectionsEnabled", True)
        settings.set("/rtx/flow/rayTracedTranslucencyEnabled", True)

    def _create_flow(self):
        """Create flow effect using Flow preset."""
        # Create root xform for positioning and scaling
        root_xform = UsdGeom.Xform.Define(self.stage, self.prim_path)
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*self.position))
        root_xform.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
        root_xform.AddScaleOp().Set(Gf.Vec3f(self.scale, self.scale, self.scale))
        root_xform.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble, "unitsResolve").Set(
            90.0
        )
        root_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "unitsResolve").Set(
            Gf.Vec3d(0.01, 0.01, 0.01)
        )
        print(f"✓ Created flow root transform at {self.prim_path}")

        # Create Fire preset as base
        success, created_prims = omni.kit.commands.execute(
            "FlowCreatePresetsCommand",
            preset_name="Fire",
            paths=[self.prim_path],
            create_copy=True,
            layer=self.layer,
            url="",
        )

        if not success:
            print(f"❌ Failed to create Flow preset at {self.prim_path}")
            return
        else:
            print(f"✓ Flow preset created successfully at {self.prim_path}")

        # Wait for preset to load
        time.sleep(0.3)

        # Configure flow parameters
        root_layer = self.stage.GetRootLayer()

        with Usd.EditContext(self.stage, root_layer):
            self._configure_emitter()
            self._configure_simulation()
            self._configure_advection()
            self._configure_vorticity()
            self._configure_colormap()
            self._configure_rendering()

        print(f"\n{'=' * 60}")
        print(f"✓ {self.flow_type.upper()} created successfully!")
        print(f"{'=' * 60}\n")

    def _configure_emitter(self):
        """Configure emitter parameters."""
        emitter_path = f"{self.prim_path}/flowEmitterSphere"
        emitter = self.stage.GetPrimAtPath(emitter_path)

        if emitter.IsValid():
            print("\nConfiguring emitter:")
            emitter.GetAttribute("smoke").Set(float(self.smoke_emit))
            emitter.GetAttribute("coupleRateSmoke").Set(float(self.couple_rate_smoke))
            emitter.GetAttribute("fuel").Set(float(self.fuel_emit))
            emitter.GetAttribute("coupleRateFuel").Set(float(self.couple_rate_fuel))
            emitter.GetAttribute("temperature").Set(float(self.temperature_emit))
            emitter.GetAttribute("coupleRateTemperature").Set(
                float(self.couple_rate_temperature)
            )
            emitter.GetAttribute("burn").Set(float(self.burn_emit))
            emitter.GetAttribute("coupleRateBurn").Set(float(self.couple_rate_burn))
            emitter.GetAttribute("velocity").Set(self.velocity)
            emitter.GetAttribute("coupleRateVelocity").Set(
                float(self.couple_rate_velocity)
            )
            emitter.GetAttribute("velocityIsWorldSpace").Set(False)
            emitter.GetAttribute("divergence").Set(float(self.divergence_emit))
            emitter.GetAttribute("coupleRateDivergence").Set(
                float(self.couple_rate_divergence)
            )
            emitter.GetAttribute("radius").Set(float(self.emitter_radius))
            emitter.GetAttribute("radiusIsWorldSpace").Set(True)
            emitter.GetAttribute("enabled").Set(True)
            print("  ✓ Emitter configured")

    def _configure_simulation(self):
        """Configure simulation parameters."""
        simulate_path = f"{self.prim_path}/flowSimulate"
        simulate = self.stage.GetPrimAtPath(simulate_path)

        if simulate.IsValid():
            print("Configuring simulation:")
            simulate.GetAttribute("densityCellSize").Set(float(self.cell_size))
            simulate.GetAttribute("layer").Set(self.layer)
            print(f"  ✓ Simulation configured (cellSize={self.cell_size})")

    def _configure_advection(self):
        """Configure advection parameters."""
        advection_path = f"{self.prim_path}/flowSimulate/advection"
        advection = self.stage.GetPrimAtPath(advection_path)

        if advection.IsValid():
            print("Configuring advection:")
            advection.GetAttribute("buoyancyPerSmoke").Set(
                float(self.buoyancy_per_smoke)
            )
            advection.GetAttribute("buoyancyPerTemp").Set(float(self.buoyancy_per_temp))
            advection.GetAttribute("buoyancyMaxSmoke").Set(1.0)
            advection.GetAttribute("combustionEnabled").Set(self.combustion_enabled)
            advection.GetAttribute("burnPerTemp").Set(float(self.burn_per_temp))
            advection.GetAttribute("ignitionTemp").Set(float(self.ignition_temp))
            advection.GetAttribute("tempPerBurn").Set(float(self.temp_per_burn))
            advection.GetAttribute("smokePerBurn").Set(float(self.smoke_per_burn))
            advection.GetAttribute("fuelPerBurn").Set(float(self.fuel_per_burn))
            advection.GetAttribute("divergencePerBurn").Set(
                float(self.divergence_per_burn)
            )
            advection.GetAttribute("coolingRate").Set(float(self.cooling_rate))
            advection.GetAttribute("gravity").Set(self.gravity)
            print("  ✓ Advection configured")

            # Configure channels
            smoke_channel = self.stage.GetPrimAtPath(f"{advection_path}/smoke")
            if smoke_channel.IsValid():
                smoke_channel.GetAttribute("damping").Set(float(self.smoke_damping))
                smoke_channel.GetAttribute("fade").Set(float(self.smoke_fade))
                print(f"  ✓ Smoke channel (fade={self.smoke_fade})")

            vel_channel = self.stage.GetPrimAtPath(f"{advection_path}/velocity")
            if vel_channel.IsValid():
                vel_channel.GetAttribute("damping").Set(float(self.velocity_damping))
                vel_channel.GetAttribute("fade").Set(float(self.velocity_fade))

    def _configure_vorticity(self):
        """Configure vorticity parameters."""
        vorticity_path = f"{self.prim_path}/flowSimulate/vorticity"
        vorticity = self.stage.GetPrimAtPath(vorticity_path)

        if vorticity.IsValid():
            print("Configuring vorticity:")
            vorticity.GetAttribute("enabled").Set(self.vorticity_enabled)
            vorticity.GetAttribute("forceScale").Set(float(self.vorticity_force))
            vorticity.GetAttribute("constantMask").Set(0.5)
            vorticity.GetAttribute("velocityMask").Set(1.0)
            vorticity.GetAttribute("velocityLogScale").Set(1.0)
            print(f"  ✓ Vorticity configured (forceScale={self.vorticity_force})")

    def _configure_colormap(self):
        """Configure colormap."""
        colormap_path = f"{self.prim_path}/flowOffscreen/colormap"
        colormap = self.stage.GetPrimAtPath(colormap_path)

        if colormap.IsValid():
            print("Configuring colormap:")
            colormap.GetAttribute("rgbaPoints").Set(self.rgba_points)
            colormap.GetAttribute("colorScale").Set(2.5)
            colormap.GetAttribute("xPoints").Set([0.0, 0.05, 0.15, 0.6, 0.85, 1.0])
            colormap.GetAttribute("colorScalePoints").Set(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
            print("  ✓ Colormap configured")

    def _configure_rendering(self):
        """Configure rendering parameters."""
        shadow_path = f"{self.prim_path}/flowOffscreen/shadow"
        shadow = self.stage.GetPrimAtPath(shadow_path)

        if shadow.IsValid():
            shadow.GetAttribute("attenuation").Set(float(self.attenuation))
            shadow.GetAttribute("minIntensity").Set(0.125)
            print(f"  ✓ Shadow configured (attenuation={self.attenuation})")

        raymarch_path = f"{self.prim_path}/flowRender/rayMarch"
        raymarch = self.stage.GetPrimAtPath(raymarch_path)

        if raymarch.IsValid():
            raymarch.GetAttribute("attenuation").Set(float(self.raymarch_attenuation))
            print(f"  ✓ Ray march configured (attenuation={self.raymarch_attenuation})")

    def reset(self):
        """Reset flow parameters (optional, can implement randomization here)."""
        pass
