# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import collections
import numpy as np
import pathlib
import torch
from collections.abc import Callable
from typing import Any
import os

import onnxruntime as ort
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model

from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.wbc_base_policy import (
    WBCPolicy,
)
from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.Go2WBC.utils import (
    get_gravity_orientation,
    load_config,
)


class Go2WBCPolicy(WBCPolicy):
    """Simple Go2 quadruped policy using trained neural network."""

    def __init__(self, wbc_config: dict, num_envs: int = 1):
        """Initialize Go2WBCPolicy.

        Args:
            wbc_config: WBC configuration
            num_envs: Number of environments
        """
        parent_dir = pathlib.Path(__file__).parent.parent
        config_path = wbc_config.policy_config_path
        model_path = wbc_config.wbc_model_path
        self.config = load_config(config_path)
        self.wbc_config = wbc_config
        self.num_envs = num_envs

        model_path_local = retrieve_file_path(model_path, force_download=True)
        full_model_path = (
            str(parent_dir / model_path_local)
            if not os.path.isabs(model_path_local)
            else model_path_local
        )

        # Auto-detect model format and load accordingly
        if full_model_path.endswith(".onnx"):
            self.policy = self.load_onnx_policy(full_model_path)
        elif full_model_path.endswith(".pt") or full_model_path.endswith(".pth"):
            self.policy = self.load_torchscript_policy(full_model_path)
        else:
            raise ValueError(
                f"Unsupported model format. Expected .onnx or .pt/.pth, got: {full_model_path}"
            )

        # Initialize observation history buffer
        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(
            (self.num_envs, self.config["num_obs"]), dtype=np.float32
        )

        # Initialize state variables
        self.use_policy_action = True
        self.action = np.zeros(
            (self.num_envs, self.config["num_actions"]), dtype=np.float32
        )
        self.target_dof_pos = np.array(self.config["default_angles"], dtype=np.float32)
        # Initialize command arrays with proper shape
        # cmd_init should be [vx, vy, base_height, yaw_rate] (4 values)
        cmd_init = self.config["cmd_init"]
        if isinstance(cmd_init, list):
            cmd_init = np.array(cmd_init, dtype=np.float32)

        # Ensure cmd_init has 4 values: [vx, vy, base_height, yaw_rate]
        if cmd_init.ndim == 1:
            assert len(cmd_init) >= 4, (
                f"cmd_init should have 4 values [vx, vy, base_height, yaw_rate], got {len(cmd_init)}"
            )
            # Extract [vx, vy] for velocity command
            self.cmd = np.tile(cmd_init[:2], (self.num_envs, 1))  # shape: (num_envs, 2)
            # Extract base_height (index 2)
            self.height_cmd = float(cmd_init[2])
            # Extract yaw_rate (index 3)
            self.yaw_rate_cmd = float(cmd_init[3])
        else:
            # If cmd_init is 2D, extract from first row
            assert cmd_init.shape[1] >= 4, (
                f"cmd_init should have 4 values per row [vx, vy, base_height, yaw_rate], got {cmd_init.shape[1]}"
            )
            self.cmd = (
                cmd_init[:, :2]
                if cmd_init.shape[0] == self.num_envs
                else np.tile(cmd_init[0, :2], (self.num_envs, 1))
            )
            self.height_cmd = (
                float(cmd_init[0, 2])
                if cmd_init.shape[0] == self.num_envs
                else float(cmd_init[0, 2])
            )
            self.yaw_rate_cmd = (
                float(cmd_init[0, 3])
                if cmd_init.shape[0] == self.num_envs
                else float(cmd_init[0, 3])
            )

        self.freq_cmd = float(self.config["freq_cmd"])
        self.gait_indices = torch.zeros((self.num_envs, 1), dtype=torch.float32)
        # Control dt: training uses 50Hz control (dt=0.02s) with 200Hz physics (dt=0.005s) and decimation=4
        # This should match the control frequency at which compute_observation is called
        self.control_dt = self.config.get("control_dt", 0.02)  # Default to 0.02s (50Hz)

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the policy.

        Args:
            env_ids: The environment ids to reset (optional)
        """
        self.gait_indices = torch.zeros((self.num_envs, 1), dtype=torch.float32)
        # Initialize observation history buffer
        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(
            (self.num_envs, self.config["num_obs"]), dtype=np.float32
        )

        # Initialize state variables
        self.use_policy_action = True
        self.action = np.zeros(
            (self.num_envs, self.config["num_actions"]), dtype=np.float32
        )
        self.target_dof_pos = np.array(self.config["default_angles"], dtype=np.float32)
        # Initialize command arrays with proper shape
        # cmd_init should be [vx, vy, base_height, yaw_rate] (4 values)
        cmd_init = self.config["cmd_init"]
        if isinstance(cmd_init, list):
            cmd_init = np.array(cmd_init, dtype=np.float32)

        # Ensure cmd_init has 4 values: [vx, vy, base_height, yaw_rate]
        if cmd_init.ndim == 1:
            assert len(cmd_init) >= 4, (
                f"cmd_init should have 4 values [vx, vy, base_height, yaw_rate], got {len(cmd_init)}"
            )
            # Extract [vx, vy] for velocity command
            self.cmd = np.tile(cmd_init[:2], (self.num_envs, 1))  # shape: (num_envs, 2)
            # Extract base_height (index 2)
            self.height_cmd = float(cmd_init[2])
            # Extract yaw_rate (index 3)
            self.yaw_rate_cmd = float(cmd_init[3])
        else:
            # If cmd_init is 2D, extract from first row
            assert cmd_init.shape[1] >= 4, (
                f"cmd_init should have 4 values per row [vx, vy, base_height, yaw_rate], got {cmd_init.shape[1]}"
            )
            self.cmd = (
                cmd_init[:, :2]
                if cmd_init.shape[0] == self.num_envs
                else np.tile(cmd_init[0, :2], (self.num_envs, 1))
            )
            self.height_cmd = (
                float(cmd_init[0, 2])
                if cmd_init.shape[0] == self.num_envs
                else float(cmd_init[0, 2])
            )
            self.yaw_rate_cmd = (
                float(cmd_init[0, 3])
                if cmd_init.shape[0] == self.num_envs
                else float(cmd_init[0, 3])
            )

        self.freq_cmd = float(self.config["freq_cmd"])

    def load_onnx_policy(
        self, model_path: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Load the ONNX policy from the model path.

        Args:
            model_path: The path to the ONNX policy model

        Returns:
            The ONNX policy model runnable for forward pass.
        """
        model = ort.InferenceSession(model_path)

        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cpu")

        print(f"Successfully loaded ONNX policy from {model_path}")

        return run_inference

    def load_torchscript_policy(
        self, model_path: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Load the TorchScript policy from the model path.

        Args:
            model_path: The path to the TorchScript (.pt or .pth) policy model

        Returns:
            The TorchScript policy model runnable for forward pass.
        """
        model = load_torchscript_model(model_path, device="cpu")

        def run_inference(input_tensor):
            with torch.no_grad():
                # Ensure input is on CPU and in correct format
                if isinstance(input_tensor, np.ndarray):
                    input_tensor = torch.from_numpy(input_tensor).float()
                output = model(input_tensor)
                return output

        print(f"Successfully loaded TorchScript policy from {model_path}")

        return run_inference

    def compute_observation(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        """Compute the observation vector from current state"""
        # Update gait indices for quadruped locomotion
        # Use control_dt (not physics_dt) since this is called at control frequency
        # Training: control_dt = 0.02s (50Hz control with 200Hz physics, decimation=4)
        self.gait_indices = torch.remainder(
            self.gait_indices + self.control_dt * self.freq_cmd, 1.0
        )
        durations = torch.full_like(self.gait_indices, 0.5)
        phases = [0.0, 0.5, 0.25, 0.75]  # FL, FR, RL, RR
        foot_indices = [self.gait_indices + phases[i] for i in range(4)]
        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
        )
        for fi in foot_indices:
            stance = fi < durations
            swing = fi >= durations
            fi[stance] = fi[stance] * (0.5 / durations[stance])
            fi[swing] = 0.5 + (fi[swing] - durations[swing]) * (
                0.5 / (1 - durations[swing])
            )

        # Compute clock inputs: shape should be (num_envs, 4)
        # Each foot_indices[i] has shape (num_envs, 1), so we concatenate along dim=1
        clock_sin = [torch.sin(2 * np.pi * fi) for fi in foot_indices]
        self.clock_inputs = torch.cat(clock_sin, dim=1)  # (num_envs, 4)

        body_indices = list(self.wbc_config.body_ids)

        n_joints = len(body_indices)

        # Extract joint data
        assert self.num_envs == observation["q"].shape[0]
        qj = observation["q"][:, body_indices].copy()
        dqj = observation["dq"][:, body_indices].copy()

        # Extract floating base data
        quat = observation["floating_base_pose"][:, 3:7].copy()  # quaternion
        omega = observation["floating_base_vel"][:, 3:6].copy()  # angular velocity

        # Handle default angles - ensure it matches the number of joints
        default_angles_list = self.config["default_angles"]
        if len(default_angles_list) < n_joints:
            # Pad with zeros if default_angles is shorter than n_joints
            padded_defaults = np.zeros(n_joints, dtype=np.float32)
            padded_defaults[: len(default_angles_list)] = np.array(
                default_angles_list, dtype=np.float32
            )
        else:
            # Use first n_joints elements if default_angles is longer
            padded_defaults = np.array(default_angles_list[:n_joints], dtype=np.float32)

        # Ensure padded_defaults has shape (n_joints,) for broadcasting
        if padded_defaults.ndim == 0:
            padded_defaults = padded_defaults.reshape(1)
        if padded_defaults.shape[0] != n_joints:
            raise ValueError(
                f"Default angles shape mismatch: expected {n_joints} joints, "
                f"got {padded_defaults.shape[0]} from {len(default_angles_list)} default angles"
            )

        # Match training observation format (no scaling applied)
        # joint_pos_rel = joint_pos - default_joint_pos (no scaling)
        # qj has shape (num_envs, n_joints), padded_defaults has shape (n_joints,)
        # Broadcasting will work correctly
        qj_rel = qj - padded_defaults
        # joint_vel_rel = joint_vel - default_joint_vel (default_joint_vel is usually 0, so just joint_vel)
        dqj_rel = dqj  # Assuming default_joint_vel is 0

        # projected_gravity: gravity vector in body frame (no scaling)
        gravity_orientation = get_gravity_orientation(quat)

        # base_ang_vel: body frame angular velocity (no scaling)
        base_ang_vel = omega

        # Calculate single observation dimension
        # Match training observation order: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
        # + velocity_commands(3) + height_commands(1) + joint_pos_rel(12) + joint_vel_rel(12) + actions(12) = 49
        single_obs_dim = 3 + 3 + 3 + 3 + 1 + n_joints + n_joints + n_joints

        # Extract base linear velocity from observation (already in body frame, no scaling)
        base_lin_vel = observation["floating_base_vel"][:, :3].copy()  # (num_envs, 3)

        # Create single observation matching training order (exactly as unified_obs_cfg.py)
        single_obs = np.zeros((self.num_envs, single_obs_dim), dtype=np.float32)

        # 1. base_lin_vel: 3 (body frame linear velocity, no scaling)
        single_obs[:, 0:3] = base_lin_vel

        # 2. base_ang_vel: 3 (body frame angular velocity, no scaling)
        single_obs[:, 3:6] = base_ang_vel

        # 3. projected_gravity: 3 (gravity vector in body frame, no scaling)
        single_obs[:, 6:9] = gravity_orientation

        # 4. velocity_commands: 3 (vx, vy, yaw_rate from base_velocity command, no scaling)
        # base_velocity_from_command extracts cmd[:, :3] directly without scaling
        single_obs[:, 9:11] = self.cmd[:, :2]  # vx, vy (no scaling)
        yaw_rate_val = (
            self.yaw_rate_cmd[0]
            if isinstance(self.yaw_rate_cmd, (np.ndarray, list))
            else self.yaw_rate_cmd
        )
        single_obs[:, 11:12] = np.full(
            (self.num_envs, 1), yaw_rate_val
        )  # yaw_rate (no scaling)

        # 5. height_commands: 1 (from go2_unified_height_command, no scaling)
        single_obs[:, 12:13] = np.full((self.num_envs, 1), self.height_cmd)

        # 6. joint_pos_rel: 12 (joint_pos - default_joint_pos, no scaling)
        single_obs[:, 13 : 13 + n_joints] = qj_rel

        # 7. joint_vel_rel: 12 (joint_vel - default_joint_vel, no scaling, default_joint_vel is 0)
        single_obs[:, 13 + n_joints : 13 + 2 * n_joints] = dqj_rel

        # 8. actions: 12 (last action, no scaling)
        single_obs[:, 13 + 2 * n_joints : 13 + 3 * n_joints] = self.action

        return single_obs, single_obs_dim

    def set_observation(self, observation: dict[str, Any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing single observation from current state
                        Should include 'obs' key with current single observation
        """

        # Extract the single observation
        self.observation = observation
        single_obs, single_obs_dim = self.compute_observation(observation)

        # Check if model expects history or single frame
        # If num_obs matches single_obs_dim, use single frame (no history)
        # Otherwise, use history
        if self.config["num_obs"] == single_obs_dim:
            # Model expects single frame observation (no history)
            self.obs_tensor = torch.from_numpy(single_obs)
        else:
            # Model expects history - add current observation to history
            self.obs_history.append(single_obs)

            # Fill history with zeros if not enough observations yet
            while len(self.obs_history) < self.config["obs_history_len"]:
                self.obs_history.appendleft(np.zeros_like(single_obs))

            # Construct full observation with history
            single_obs_dim = single_obs.shape[1]
            for i, hist_obs in enumerate(self.obs_history):
                start_idx = i * single_obs_dim
                end_idx = start_idx + single_obs_dim
                self.obs_buffer[:, start_idx:end_idx] = hist_obs

            # Convert to tensor for policy
            self.obs_tensor = torch.from_numpy(self.obs_buffer)

        assert self.obs_tensor.shape[1] == self.config["num_obs"], (
            f"Observation dimension mismatch: expected {self.config['num_obs']}, "
            f"got {self.obs_tensor.shape[1]}"
        )

        # Print observation input for debugging
        if not hasattr(self, "_frame_count"):
            self._frame_count = 0
        self._frame_count += 1

        obs_np = (
            self.obs_tensor.numpy()
            if isinstance(self.obs_tensor, torch.Tensor)
            else self.obs_tensor
        )
        print(f"\n{'=' * 80}")
        print(f"Frame {self._frame_count} - Observation Input (first env):")
        print(f"{'=' * 80}")
        print(f"Shape: {obs_np.shape}")
        print("\nObservation breakdown (first env):")
        print(f"  [0:3]   base_lin_vel:      {obs_np[0, 0:3]}")
        print(f"  [3:6]   base_ang_vel:      {obs_np[0, 3:6]}")
        print(f"  [6:9]   projected_gravity: {obs_np[0, 6:9]}")
        print(f"  [9:12]  velocity_commands: {obs_np[0, 9:12]}  # [vx, vy, yaw_rate]")
        print(f"  [12:13] height_commands:   {obs_np[0, 12:13]}")
        print(f"  [13:25] joint_pos_rel:     {obs_np[0, 13:25]}")
        print(f"  [25:37] joint_vel_rel:     {obs_np[0, 25:37]}")
        print(f"  [37:49] actions (last):    {obs_np[0, 37:49]}")
        print(f"{'=' * 80}")

    def set_goal(self, goal: dict[str, Any]):
        """Set the goal for the policy.

        Args:
            goal: Dictionary containing the goal for the policy
        """
        print("goal: ", goal)
        if "velocity_cmd" in goal:
            velocity_cmd = goal["velocity_cmd"]
            # Ensure velocity_cmd has shape (num_envs, 2) for [vx, vy]
            if isinstance(velocity_cmd, np.ndarray):
                if velocity_cmd.ndim == 1:
                    self.cmd = np.tile(velocity_cmd[:2], (self.num_envs, 1))
                else:
                    self.cmd = (
                        velocity_cmd[:, :2]
                        if velocity_cmd.shape[1] >= 2
                        else np.tile(velocity_cmd[0, :2], (self.num_envs, 1))
                    )
            else:
                # Convert to numpy array
                velocity_cmd = np.array(velocity_cmd, dtype=np.float32)
                if velocity_cmd.ndim == 1:
                    self.cmd = np.tile(velocity_cmd[:2], (self.num_envs, 1))
                else:
                    self.cmd = (
                        velocity_cmd[:, :2]
                        if velocity_cmd.shape[1] >= 2
                        else np.tile(velocity_cmd[0, :2], (self.num_envs, 1))
                    )

        if "base_height_command" in goal:
            height_cmd = goal["base_height_command"]
            if isinstance(height_cmd, (list, np.ndarray)):
                if isinstance(height_cmd, np.ndarray) and height_cmd.ndim > 0:
                    self.height_cmd = float(height_cmd[0])
                else:
                    self.height_cmd = float(
                        height_cmd[0]
                        if len(height_cmd) > 0
                        else self.config["height_cmd"]
                    )
            else:
                self.height_cmd = float(height_cmd)

        if "yaw_rate_cmd" in goal:
            yaw_rate_val = goal["yaw_rate_cmd"]
            if isinstance(yaw_rate_val, (list, np.ndarray)):
                if isinstance(yaw_rate_val, np.ndarray) and yaw_rate_val.ndim > 0:
                    self.yaw_rate_cmd = float(yaw_rate_val[0])
                else:
                    self.yaw_rate_cmd = float(
                        yaw_rate_val[0] if len(yaw_rate_val) > 0 else 0.0
                    )
            else:
                self.yaw_rate_cmd = float(yaw_rate_val)

    def get_action(self) -> dict[str, Any]:
        """Compute and return the next action based on current observation.

        Returns:
            Dictionary containing the action to be executed
        """
        if self.obs_tensor is None:
            raise ValueError("No observation set. Call set_observation() first.")

        # Run policy inference
        with torch.no_grad():
            # Ensure input tensor has correct shape and dtype
            if isinstance(self.obs_tensor, np.ndarray):
                obs_input = torch.from_numpy(self.obs_tensor).float()
            else:
                obs_input = self.obs_tensor.float()

            # Check observation shape
            if obs_input.shape[1] != self.config["num_obs"]:
                raise ValueError(
                    f"Observation shape mismatch: expected (num_envs, {self.config['num_obs']}), "
                    f"got {obs_input.shape}"
                )

            # Run policy for all environments
            policy_action = self.policy(obs_input)

            # Convert to numpy if needed
            if isinstance(policy_action, torch.Tensor):
                policy_action = policy_action.detach().cpu().numpy()

            # Check policy output shape
            if policy_action.shape != (self.num_envs, self.config["num_actions"]):
                raise ValueError(
                    f"Policy output shape mismatch: expected ({self.num_envs}, {self.config['num_actions']}), "
                    f"got {policy_action.shape}"
                )

            # Store the raw policy action (before scaling) for next observation
            # This is the action that will be used in the next frame's observation
            self.action = policy_action.copy()

            # Convert to joint position targets: action * scale + default_angles
            default_angles_array = np.array(
                self.config["default_angles"], dtype=np.float32
            )
            if default_angles_array.ndim == 1:
                # Ensure default_angles has correct shape (num_envs, num_actions)
                default_angles_array = np.tile(default_angles_array, (self.num_envs, 1))

            # Apply scaling and offset: target = action * scale + default_angles
            cmd_q = policy_action * self.config["action_scale"] + default_angles_array

            # Print policy output for debugging
            print(f"\n{'=' * 80}")
            print(f"Frame {self._frame_count} - Policy Output:")
            print(f"{'=' * 80}")
            print(f"Policy action shape: {policy_action.shape}")
            print(f"Policy action (first env, all 12): {policy_action[0, :]}")
            print(f"Action scale: {self.config['action_scale']}")
            print(f"Default angles (first env, all 12): {default_angles_array[0, :]}")
            print(f"Target joint pos (first env, all 12): {cmd_q[0, :]}")
            print(
                f"Command (vx, vy, height, yaw): [{self.cmd[0, 0]:.3f}, {self.cmd[0, 1]:.3f}, {self.height_cmd:.3f}, {self.yaw_rate_cmd:.3f}]"
            )
            print(f"{'=' * 80}\n")

        return cmd_q
        # return default_angles_array
