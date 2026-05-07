"""Handover command: issues the bimanual ``Handover`` atomic skill.

The atomic skill (Collect/AtomicSkill/Handover.py) drives both arms
through the full state machine — left grasps + lifts, paired IK picks
a right-arm-friendly handover pose, right takes over, left releases.
This command just wraps that as a top-level task so the AutoCollect
framework can schedule it like any other task.
"""

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from magicsim.Collect.Command.Task import Task
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class Handover(Task):
    """Top-level wrapper around the Handover atomic skill."""

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.robot_id = int(OmegaConf.select(config, "robot_id", default=0))
        self.obj_type = OmegaConf.select(config, "obj_type", default="rigid")
        self.obj_name = OmegaConf.select(config, "obj_name", default="mug")
        self.obj_id = int(OmegaConf.select(config, "obj_id", default=0))

        self.init_count = 0
        self.init_limit = int(OmegaConf.select(config, "init_limit", default=50))
        self.max_attempt = int(OmegaConf.select(config, "max_attempt", default=2))
        self.attempt_count = 0

    def reset(self):
        self.current_state = "ready"
        self.current_action = None
        self.last_action = None
        self.init_count = 0
        self.attempt_count = 0

    def _handover_action(self) -> list:
        # AtomicSkill Handover.reset signature:
        #   ["Handover", robot_id, obj_type, obj_name, obj_id]
        return [
            "Handover",
            self.robot_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
        ]

    def step(self):
        if self.current_state == "failed: task max attempt reached":
            return "Failed"
        # Atomic finished its full state machine — request explicit env
        # reset. Returning "Failed" puts this env_id in failed_env_ids
        # downstream → ``TaskBaseEnv.step`` flips truncated=1 and calls
        # ``reset_idx`` → mug back on table, robots back to home pose.
        # Without this, ``get_termination`` is the only reset trigger,
        # and the geometric thresholds (left_release / right_close /
        # mug_z) are easy to miss if the controller times out short of
        # convergence — env then drifts into a wonky state and the next
        # Handover attempt starts from there.
        if self.current_state == "atomic done: request reset":
            return "Failed"
        if self.init_count < self.init_limit:
            self.init_count += 1
            self.current_state = "initializing"
            self.current_action = None
            self.last_action = None
            return None
        self.current_state = "running"
        self.current_action = self._handover_action()
        self.last_action = None
        if self.attempt_count >= self.max_attempt:
            self.current_state = "failed: task max attempt reached"
            return "Failed"
        return self.current_action

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "type": "Handover",
            "last_action": self.last_action,
            "current_action": self.current_action,
        }
        if self.current_state == "initializing":
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
        if self.current_state == "failed: task max attempt reached":
            return {
                **base,
                "finished": False,
                "state": self.current_state,
                "truncated": 5,
            }

        atomic = info["atomic_skill_info"][self.env_id]
        env_terminated = info["env_info"][2][self.env_id]

        # The frame AFTER we returned "Failed" from step(): env reset
        # has already fired, AtomicSkillManager cleared the skill, so
        # ``atomic`` is None here. Mark the task done so AutoCollect
        # clears it and spawns a fresh Handover next frame on the new
        # env state.
        if atomic is None:
            self.current_state = "task done: env reset complete"
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }

        if atomic["finished"]:
            # Atomic skill ran the full 11-phase Handover state machine
            # — that's the unit of work, regardless of whether
            # ``env_terminated`` fired. ALWAYS request env reset by
            # transitioning to the ``atomic done: request reset`` state;
            # the next ``step()`` returns "Failed", which puts this env
            # in ``failed_env_ids`` → ``TaskBaseEnv.step`` flips
            # ``truncated=1`` and calls ``reset_idx``. After the reset,
            # the task is cleared (truncated path below) and a brand-new
            # Handover task spawns on the fresh env.
            #
            # The retry loop (attempt_count → max_attempt) was removed:
            # it was respawning Handover atomic skills on top of the
            # PREVIOUS final state — right hand still holding the mug,
            # left hand wherever it ended up — and the new IK started
            # from that wonky configuration.
            self.current_state = "atomic done: request reset"
            self.attempt_count += 1
            return {
                **base,
                "finished": False,
                "state": (
                    "success: env terminated, requesting reset"
                    if env_terminated
                    else "atomic done (env_terminated=False), requesting reset"
                ),
                "truncated": 0,
                "attempt_count": self.attempt_count,
            }

        trunc = atomic["truncated"]
        if trunc == 1:
            self.current_state = "success: env terminated first"
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        if trunc == 2:
            self.current_state = "truncated: env truncated first"
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": 2,
            }
        if trunc in (3, 4):
            # Surface as FINISHED so AutoCollect resets the env between attempts.
            self.current_state = (
                "failed: global planner failed to plan"
                if trunc == 3
                else "truncated: atomic skill failed to plan"
            )
            return {
                **base,
                "finished": True,
                "state": self.current_state,
                "truncated": trunc,
            }

        self.current_state = "running"
        return {**base, "finished": False, "state": self.current_state, "truncated": 0}
