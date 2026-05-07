from isaacsim.core.utils.semantics import add_labels
from omegaconf import DictConfig
import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, Gf
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np
import omni.anim.graph.core as ag
import carb
from typing import List, Tuple
from isaacsim.storage.native import get_assets_root_path
from omni.anim.people.python_ext import get_instance
from omni.anim.people.scripts.custom_command.populate_anim_graph import (
    populate_anim_graph,
)
from omni.anim.people.scripts.custom_command.command_templates import TimingTemplate
from omni.anim.people.scripts.custom_command.defines import CustomCommandTemplate
from omni.metropolis.utils.carb_util import CarbUtil


class SimpleIdle:
    """Simplified Idle command - idle in place using set_variable only"""

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
    ):
        self.character = character
        self.command = command
        self.character_name = character_name
        self.duration = float(command[1]) if len(command) > 1 else 5.0
        self.idle_time = 0
        self.is_setup = False
        self.finished = False

    def get_command_name(self):
        return "Idle"

    def setup(self):
        self.character.set_variable("Action", "None")
        self.is_setup = True
        carb.log_info(f"{self.character_name} idling for {self.duration}s")

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.idle_time += dt
        if self.idle_time >= self.duration:
            self.finished = True
            return True
        return False

    def force_quit_command(self):
        self.character.set_variable("Action", "None")


class SimpleLookAround:
    """Simplified LookAround command - look around using lookaround variable"""

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
    ):
        self.character = character
        self.command = command
        self.character_name = character_name
        self.duration = float(command[1]) if len(command) > 1 else 5.0
        self.look_time = 0
        self.is_setup = False
        self.finished = False

    def get_command_name(self):
        return "LookAround"

    def setup(self):
        self.character.set_variable("Action", "None")  # Stay in Idle state
        self.character.set_variable("lookaround", 1.0)  # Enable lookaround
        self.is_setup = True
        carb.log_info(f"{self.character_name} looking around for {self.duration}s")

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.look_time += dt
        if self.look_time >= self.duration:
            self.character.set_variable("lookaround", 0.0)  # Disable lookaround
            self.finished = True
            return True
        return False

    def force_quit_command(self):
        self.character.set_variable("lookaround", 0.0)


class SimpleSit:
    """Simplified Sit command - sit in place using set_variable only"""

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
    ):
        self.character = character
        self.command = command
        self.character_name = character_name
        self.duration = float(command[1]) if len(command) > 1 else 5.0
        self.sit_time = 0
        self.is_setup = False
        self.finished = False

    def get_command_name(self):
        return "Sit"

    def setup(self):
        self.character.set_variable("Action", "Sit")
        self.is_setup = True
        carb.log_info(f"{self.character_name} sitting in place for {self.duration}s")

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.sit_time += dt
        if self.sit_time >= self.duration:
            self.character.set_variable("Action", "None")
            self.finished = True
            return True
        return False

    def force_quit_command(self):
        self.character.set_variable("Action", "None")


class SimpleTalk:
    """Simplified Talk command - talk in place using set_variable only"""

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
    ):
        self.character = character
        self.command = command
        self.character_name = character_name
        self.duration = float(command[1]) if len(command) > 1 else 5.0
        self.talk_time = 0
        self.is_setup = False
        self.finished = False

    def get_command_name(self):
        return "Talk"

    def setup(self):
        self.character.set_variable("Action", "Talk")
        self.is_setup = True
        carb.log_info(f"{self.character_name} talking in place for {self.duration}s")

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.talk_time += dt
        if self.talk_time >= self.duration:
            self.character.set_variable("Action", "None")
            self.finished = True
            return True
        return False

    def force_quit_command(self):
        self.character.set_variable("Action", "None")


class SimpleGoTo:
    """Simplified GoTo command using simplified navigation logic

    Accepts local coordinates, converts to global, and uses simplified path generation.
    Replicates NavigationManager's basic functionality for direct paths.
    """

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
        env_origin=None,
    ):
        self.character = character
        self.character_name = character_name
        self.env_origin = env_origin if env_origin is not None else [0.0, 0.0, 0.0]

        # Parse local coordinates
        target_x_local = float(command[1])
        target_y_local = float(command[2])
        target_z_local = float(command[3]) if len(command) > 3 else 0.0
        self.target_angle = float(command[4]) if len(command) > 4 else 0.0

        # Convert to global coordinates
        local_pos = np.array([target_x_local, target_y_local, target_z_local])

        if isinstance(self.env_origin, torch.Tensor):
            env_origin_np = self.env_origin.cpu().numpy()
        elif isinstance(self.env_origin, list):
            env_origin_np = np.array(self.env_origin)
        else:
            env_origin_np = np.array(self.env_origin)

        self.target_pos_global = local_pos + env_origin_np

        # Replicate NavigationManager's path management
        self.path_points = []  # Path for animation
        self.path_targets = []  # Target positions
        self.arrival_threshold = 0.25  # Distance to consider arrived

        # Walk state
        self.is_setup = False
        self.finished = False
        self.desired_walk_speed = 1.0
        self.actual_walk_speed = 0.0

        carb.log_info(
            f"{character_name} GoTo: local({target_x_local:.2f}, {target_y_local:.2f}) -> global({self.target_pos_global[0]:.2f}, {self.target_pos_global[1]:.2f})"
        )

    def get_command_name(self):
        return "GoTo"

    def setup(self):
        # Get start position
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        self.character.get_world_transform(pos, rot)
        start_pos = np.array([pos.x, pos.y, pos.z])

        # Generate simple path (replicate NavigationManager.generate_path with navmesh_enabled=False)
        self.path_points = [
            carb.Float3(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])),
            carb.Float3(
                float(self.target_pos_global[0]),
                float(self.target_pos_global[1]),
                float(self.target_pos_global[2]),
            ),
        ]
        self.path_targets = [self.target_pos_global]

        # Set animation variables
        self.character.set_variable("Action", "Walk")

        self.is_setup = True

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def destination_reached(self):
        """Check if destination is reached (replicate NavigationManager.destination_reached)"""
        if not self.path_targets:
            return True

        # Get current position
        pos = carb.Float3(0, 0, 0)
        rot = carb.Float4(0, 0, 0, 0)
        self.character.get_world_transform(pos, rot)
        current_pos = np.array([pos.x, pos.y, pos.z])

        # Check distance to target
        target = self.path_targets[0]
        distance = np.linalg.norm(current_pos - target)
        return distance < self.arrival_threshold

    def update(self, dt):
        # Replicate base_command.walk() logic
        if self.destination_reached():
            self.desired_walk_speed = 0.0
            if self.actual_walk_speed < 0.001:
                self.character.set_variable("Action", "None")
                self.character.set_variable("PathPoints", [])
                self.path_points = []
                self.path_targets = []
                self.finished = True
                return True
        else:
            self.desired_walk_speed = 1.0

        # Set path points for animation
        self.character.set_variable("Action", "Walk")
        self.character.set_variable("PathPoints", self.path_points)

        # Blend walking animation when starting or stopping
        max_change = dt / 0.2  # WalkBlendTime = 0.2
        delta_walk = CarbUtil.clamp(
            self.desired_walk_speed - self.actual_walk_speed,
            -1 * max_change,
            max_change,
        )
        self.actual_walk_speed = CarbUtil.clamp(
            self.actual_walk_speed + delta_walk, 0.0, 1.0
        )
        self.character.set_variable("Walk", self.actual_walk_speed)

        return False

    def force_quit_command(self):
        self.character.set_variable("Action", "None")
        self.character.set_variable("Walk", 0.0)
        self.character.set_variable("PathPoints", [])


class Avatar(SingleGeometryPrim):
    """Avatar class to manage individual avatar instances in the simulation."""

    # Custom command animation relative paths (from Isaac Sim asset root)
    CUSTOM_COMMAND_ANIMATIONS = [
        "/Isaac/People/Animations/push_button.skelanim.usd",
        "/Isaac/People/Animations/type_keyboard.skelanim.usd",
        # Add more custom animation paths here
    ]

    @staticmethod
    def setup_custom_commands(animation_paths=None):
        """Setup custom commands by loading animation USD files"""
        custom_cmd_mgr = get_instance().get_custom_command_manager()
        if not custom_cmd_mgr:
            return

        assets_root = get_assets_root_path()
        if not assets_root:
            return

        paths = animation_paths or Avatar.CUSTOM_COMMAND_ANIMATIONS

        count = 0
        for rel_path in paths:
            full_path = assets_root + rel_path
            if not custom_cmd_mgr.is_custom_command_anim_exist(full_path):
                if custom_cmd_mgr.add_custom_command(full_path):
                    count += 1

        if count > 0:
            populate_anim_graph()
            carb.log_info(f"Loaded {count} custom commands")

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        layout_info=None,
        collision: bool = True,
    ):
        # Store parameters as temporary variables before calling super().__init__()
        self.usd_path = usd_path
        self.config = config
        self.env_origin = env_origin
        self.stage = get_current_stage()
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        # Load USD reference
        prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Failed to load USD from {usd_path} to {prim_path}")

        # Get initial pose from layout_info
        if layout_info:
            pos_from_layout = layout_info["pos"]
            # Convert local (env-frame) position to world by adding env origin
            init_pos = (
                np.array(pos_from_layout, dtype=float)
                + np.array(env_origin, dtype=float)
            ).tolist()
            init_ori = layout_info.get("ori", [1.0, 0.0, 0.0, 0.0])
            init_scale = layout_info.get("scale", [1.0, 1.0, 1.0])
        else:
            init_pos = np.array(env_origin, dtype=float).tolist()
            init_ori = [1.0, 0.0, 0.0, 0.0]
            init_scale = [1.0, 1.0, 1.0]

        # Initialize SingleGeometryPrim (this will set self.prim_path as a property)
        super().__init__(
            prim_path=prim_path,
            name=prim_path.split("/")[-1],
            translation=init_pos,
            orientation=init_ori,
            scale=init_scale,
            visible=True,
            collision=collision,
        )

        self.skelroot_prim = self.get_skel_root_prim()

        # Command execution related variables
        self.character = None  # ag.Character object (from Animation Graph API)
        self.character_name = prim_path.split("/")[-1]
        self.current_command = None
        self.commands: List[Tuple] = []  # Command queue
        add_labels(self.prim, ["avatar"])
        self.is_initialized = False

    def get_skel_root_prim(self):
        for prim in Usd.PrimRange(self.prim):
            if prim.GetTypeName() == "SkelRoot":
                return prim
        else:
            raise ValueError(f"SkelRoot prim not found under {self.prim_path}")

    def set_world_pose(self, position, orientation):
        """Set world pose using Animation Graph API

        Accepts local coordinates and converts to global coordinates internally.

        Args:
            position: Position as [x, y, z] list or array (local coordinates)
            orientation: Orientation as quaternion [w, x, y, z] or [x, y, z, w]
        """
        # If character is initialized, use Animation Graph API
        if self.character is not None:
            # Convert local position to global
            local_pos = np.array(position, dtype=float)
            if isinstance(self.env_origin, torch.Tensor):
                env_origin_np = self.env_origin.cpu().numpy()
            else:
                env_origin_np = np.array(self.env_origin, dtype=float)

            global_pos = local_pos + env_origin_np

            # Convert to carb.Float3 for Animation Graph API
            pos = carb.Float3(
                float(global_pos[0]), float(global_pos[1]), float(global_pos[2])
            )

            # Convert orientation to carb.Float4 (x, y, z, w format)
            if len(orientation) == 4:
                # Assume [w, x, y, z] format, convert to [x, y, z, w]
                ori_array = np.array(orientation, dtype=float)
                rot = carb.Float4(
                    float(ori_array[1]),
                    float(ori_array[2]),
                    float(ori_array[3]),
                    float(ori_array[0]),
                )
            else:
                # Default quaternion
                rot = carb.Float4(0, 0, 0, 1)

            # Use Animation Graph API
            self.character.set_world_transform(pos, rot)
        else:
            # Fallback to GeometryPrim method if character not initialized
            # Convert to global first
            local_pos = np.array(position, dtype=float)
            if isinstance(self.env_origin, torch.Tensor):
                env_origin_np = self.env_origin.cpu().numpy()
            else:
                env_origin_np = np.array(self.env_origin, dtype=float)

            global_pos = local_pos + env_origin_np

            # Convert orientation
            if len(orientation) == 3:
                rx, ry, rz = (float(v) for v in orientation)
                quat = (
                    Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
                    * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                    * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
                )
                quat = quat.GetQuat()
                orientation_quat = np.array([quat.GetReal(), *quat.GetImaginary()])
            else:
                orientation_quat = np.array(orientation, dtype=float)

            # Use inherited method with global coordinates
            super().set_world_pose(position=global_pos, orientation=orientation_quat)

    def get_world_pose(self, local_coords=True):
        """Get world pose using Animation Graph API

        Args:
            local_coords: If True, return local coordinates; if False, return global coordinates

        Returns:
            tuple: (position, rotation) where position is [x,y,z] and rotation is [x,y,z,w]
        """
        if self.character is not None:
            # Use Animation Graph API
            pos = carb.Float3(0, 0, 0)
            rot = carb.Float4(0, 0, 0, 0)
            self.character.get_world_transform(pos, rot)

            # Convert to numpy arrays
            global_pos = np.array([pos.x, pos.y, pos.z])
            rotation = np.array([rot.x, rot.y, rot.z, rot.w])

            if local_coords:
                # Convert global to local
                if isinstance(self.env_origin, torch.Tensor):
                    env_origin_np = self.env_origin.cpu().numpy()
                else:
                    env_origin_np = np.array(self.env_origin, dtype=float)

                local_pos = global_pos - env_origin_np
                return local_pos.tolist(), rotation.tolist()
            else:
                return global_pos.tolist(), rotation.tolist()
        else:
            # Fallback if character not initialized
            return [0, 0, 0], [0, 0, 0, 1]

    def reset(self, soft: bool = False):
        """Reset avatar pose using LayoutManager."""
        if not self.layout_manager:
            raise RuntimeError(
                f"No layout_manager for avatar {self.prim_path}. Cannot reset."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            raise ValueError(
                f"Could not extract env_id for {self.prim_path}. Cannot reset."
            )

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self.prim_path, reset_type=reset_type
        )

        if not new_layout:
            raise RuntimeError(
                f"LayoutManager did not provide new layout for {self.prim_path}."
            )

        pos = new_layout["pos"]
        ori = new_layout["ori"]
        # Convert local (env-frame) position to world by adding env origin
        world_pos = (
            np.array(pos, dtype=float) + np.array(self.env_origin, dtype=float)
        ).tolist()
        self.set_world_pose(position=world_pos, orientation=ori)

    def _extract_env_id_from_prim_path(self):
        """Extract env_id from prim_path."""
        try:
            parts = self.prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def init_character(self):
        """Initialize animation graph character object (can only be called at runtime)"""
        if self.is_initialized:
            return True

        # Get animation graph character object
        self.character = ag.get_character(str(self.skelroot_prim.GetPrimPath()))
        if self.character is None:
            carb.log_warn(
                f"Cannot get character object: {self.skelroot_prim.GetPrimPath()}"
            )
            return False

        # Initialize animation variables
        self.character.set_variable("Action", "None")

        self.is_initialized = True
        carb.log_info(f"Character initialized: {self.character_name}")
        return True

    @staticmethod
    def init_custom_commands_once():
        """Initialize custom commands once (call before first avatar is created)"""
        if not hasattr(Avatar, "_custom_commands_initialized"):
            Avatar.setup_custom_commands()
            Avatar._custom_commands_initialized = True

    def inject_command(self, command_list: List, execute_immediately: bool = True):
        """
        Inject commands to the character

        Args:
            command_list: Command list, format: [["CommandType", "param1", ...], ...]
            execute_immediately: Whether to execute immediately
        """
        cmd_array = []
        for command in command_list:
            if isinstance(command, list):
                cmd_array.append((None, command))
            elif isinstance(command, str):
                words = command.strip().split()
                cmd_array.append((None, words))

        if execute_immediately:
            if self.commands and cmd_array:
                self.commands[1:1] = cmd_array
            else:
                self.commands[0:0] = cmd_array
        else:
            self.commands.extend(cmd_array)

        carb.log_info(
            f"Commands injected to {self.character_name}: {len(cmd_array)} commands"
        )

    def execute_command(self, delta_time: float):
        """Execute command queue"""
        while not self.current_command:
            if not self.commands:
                return

            next_cmd_pair = self.commands[0]
            command_id, command = next_cmd_pair

            if len(command) < 1:
                self.commands.pop(0)
                continue

            self.current_command = self._create_command_object(command_id, command)

            if self.current_command:
                carb.log_info(f"Start executing command: {command}")
            else:
                self.commands.pop(0)

        if self.current_command:
            try:
                if self.current_command.execute(delta_time):
                    carb.log_info(
                        f"Command completed: {self.current_command.get_command_name()}"
                    )
                    self.commands.pop(0)
                    self.current_command = None
            except Exception as e:
                carb.log_error(f"Command execution error: {e}")
                self.commands.pop(0)
                self.current_command = None

    def _create_command_object(self, command_id, command):
        """Create command object"""
        command_params = {
            "character": self.character,
            "command": command,
            "character_name": str(self.character_name),
            "navigation_manager": None,  # Not used by our simple commands
            "command_id": command_id,
            "update_metadata_callback_fn": self._dummy_metadata_callback,
        }

        if len(command) < 1:
            return None

        command_type = command[0]

        if command_type == "GoTo":
            # Add env_origin for local to global coordinate conversion
            command_params["env_origin"] = self.env_origin
            return SimpleGoTo(**command_params)
        elif command_type == "Idle":
            return SimpleIdle(**command_params)
        elif command_type == "LookAround":
            return SimpleLookAround(**command_params)
        elif command_type == "Sit":
            return SimpleSit(**command_params)
        elif command_type == "Talk":
            return SimpleTalk(**command_params)
        else:
            # Try to create custom command
            try:
                custom_cmd_mgr = get_instance().get_custom_command_manager()

                if command_type in custom_cmd_mgr.get_all_custom_command_names():
                    # Use custom command template
                    custom_command_item = custom_cmd_mgr.get_custom_command_by_name(
                        command_type
                    )
                    if custom_command_item.template == CustomCommandTemplate.TIMING:
                        return TimingTemplate(
                            **command_params, command_name=custom_command_item.name
                        )
                    else:
                        carb.log_warn(
                            f"Custom command template {custom_command_item.template} not supported yet"
                        )
                        return None
            except Exception as e:
                carb.log_warn(f"Failed to create custom command {command_type}: {e}")

            carb.log_warn(f"Unknown command type: {command_type}")
            return None

    def _dummy_metadata_callback(self, agent_name, data_name, data_value):
        """Metadata callback (placeholder)"""
        pass

    def get_current_action(self) -> str:
        """Get current executing action name"""
        if self.current_command:
            return self.current_command.get_command_name()
        return "None"

    def get_command_queue_length(self) -> int:
        """Get number of commands in queue"""
        return len(self.commands)

    def on_update(self, current_time: float, delta_time: float):
        """Update every frame"""
        if self.character is None:
            if not self.init_character():
                return

        if self.commands:
            self.execute_command(delta_time)
