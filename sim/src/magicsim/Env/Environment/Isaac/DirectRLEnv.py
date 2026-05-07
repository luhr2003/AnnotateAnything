## TODO: This code need to be checked carefully when upgrading to new IsaacLab version


import torch

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.direct_rl_env import DirectRLEnv


# We modify some part of the direct_rl_env to support resetting specific environments
class CustomDirectRLEnv(DirectRLEnv):
    def sim_step(self):
        """
        This function is used to step the simulation backend
        ! Important Function !: simulation backend step function.
        """
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.app.update()
            # if self._sim_step_counter > 100:
            #     rgb = self.render()
            #     rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(f"whole_video/render_{self._sim_step_counter}.png", rgb)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            # if (
            #     self._sim_step_counter % self.cfg.sim.render_interval == 0
            #     and is_rendering
            # ):
            #     self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # ! Below is the original code from DirectRLEnv.step()
        # self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        # self.reset_buf = self.reset_terminated | self.reset_time_outs
        # self.reward_buf = self._get_rewards() # modify to speed up

        # # -- reset envs that terminated/timed-out and log the episode information
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self._reset_idx(reset_env_ids)
        #     # update articulation kinematics
        #     self.scene.write_data_to_sim()
        #     self.sim.forward()
        #     # if sensors are added to the scene, make sure we render to reflect changes in reset
        #     if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
        #         self.sim.render()
        # ! End of original code

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        # self.obs_buf = self._get_observations()  # modified to speed up

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        # if self.cfg.observation_noise_model:
        #     self.obs_buf["policy"] = self._observation_noise_model.apply(
        #         self.obs_buf["policy"]
        #     )

        # return observations, rewards, resets and extras
        # return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

        # Only return extras
        return self.extras

    def step(self, action: torch.Tensor, env_ids: torch.Tensor) -> VecEnvStepReturn:
        """
        Compared to DirectRLEnv.step(), this function won't automatically reset environments that is successful or timed-out.
        In this fucntion, we won't call _get_dones, _get_rewards and _get_observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            The extras info of original isaaclab directenv
        """
        action = action.to(self.device)
        env_ids = env_ids.to(self.device)
        # add action noise
        # if self.cfg.action_noise_model:
        #     action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action, env_ids)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.app.update()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            # if (
            #     self._sim_step_counter % self.cfg.sim.render_interval == 0
            #     and is_rendering
            # ):
            #     self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # ! Below is the original code from DirectRLEnv.step()
        # self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        # self.reset_buf = self.reset_terminated | self.reset_time_outs
        # self.reward_buf = self._get_rewards() # modify to speed up

        # # -- reset envs that terminated/timed-out and log the episode information
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self._reset_idx(reset_env_ids)
        #     # update articulation kinematics
        #     self.scene.write_data_to_sim()
        #     self.sim.forward()
        #     # if sensors are added to the scene, make sure we render to reflect changes in reset
        #     if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
        #         self.sim.render()
        # ! End of original code

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        # self.obs_buf = self._get_observations()  # modified to speed up

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        # if self.cfg.observation_noise_model:
        #     self.obs_buf["policy"] = self._observation_noise_model.apply(
        #         self.obs_buf["policy"]
        #     )

        # return observations, rewards, resets and extras
        # return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

        # Only return extras
        return self.extras

    def get_observations(self):
        return self._get_observations()

    def get_dones(self):
        return self._get_dones()
