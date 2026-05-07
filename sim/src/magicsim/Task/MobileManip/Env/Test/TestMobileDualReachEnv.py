"""Dual-arm reach smoke test — drives each hand to its assigned cube.

Auto-derives per-term action dims by iterating
``env.scene.planner_manager.single_action_space`` — which reflects the
**planner input** layout (e.g. p_controller's 15-dim base input for x7s /
vega), not the raw joint-space dims exposed by ``robot_manager``. The test
therefore works for any ``MobileManip`` robot wired into
``MobileDualReachEnv`` without hardcoding base / arm / eef widths.
"""

from magicsim.Task.MobileManip.Env.MobileDualReachEnv import MobileDualReachEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
import torch


# Pink IK dual-arm layout: ``[ right_pose(7) | left_pose(7) ]``.
_POSE_DIM = 7
_DUAL_POSE_DIM = 2 * _POSE_DIM


def _term_dim(space) -> int:
    """Extract the feature dim from a gym Box / nested space."""
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"Cannot determine dim for space {space!r}")
    if len(shape) == 1:
        return int(shape[0])
    # Box(low=[[lo...], [hi...]]) style — second axis is the per-env feature dim.
    return int(shape[-1])


def _build_reach_action(
    env: MobileDualReachEnv, red_pose: torch.Tensor, blue_pose: torch.Tensor
) -> dict:
    """Build the dict-of-dict action consumed by ``planner_manager.step``.

    * ``base_action`` → all-NaN row. The p-controller helpers (vega / x7s)
      treat an all-NaN row as ``lock_skip`` with the last-target fallback,
      so the mobile base holds its pose while the arms reach.
    * ``arm_action``  → 14-dim ``[blue(7) | red(7)]`` for the ik_pink term
      (right hand chases blue, left hand chases red); if the term has a
      different dim it is filled with NaN instead so the smoke test still
      runs on non-dual-arm ik variants.
    * ``eef_action``  → zeros (binary open / joint_pos idle).

    Dims are pulled from ``planner_manager.single_action_space`` because
    the planner rewrites base into the 15-dim vega-style input when
    ``body.type == p_controller`` is configured — the raw
    ``robot_manager.action_managers`` dims (e.g. 3 for holonomic base)
    would underflow ``PlannerManager.total_action_dim``.
    """
    device = env.device
    n = red_pose.shape[0]
    actions: dict = {}

    planner_manager = env.scene.planner_manager
    for robot_name, robot_space in planner_manager.single_action_space.spaces.items():
        per_robot: dict = {}
        for term_name, term_space in robot_space.spaces.items():
            dim = _term_dim(term_space)
            if term_name == "base_action":
                # All-NaN → p-controller helper folds this to lock_skip and
                # falls back to the last-target (or current) base pose.
                vec = torch.full(
                    (n, dim), float("nan"), device=device, dtype=torch.float32
                )
            elif term_name == "arm_action":
                if dim == _DUAL_POSE_DIM:
                    vec = torch.cat([blue_pose, red_pose], dim=-1).to(
                        device=device, dtype=torch.float32
                    )
                else:
                    vec = torch.full(
                        (n, dim), float("nan"), device=device, dtype=torch.float32
                    )
            elif term_name == "eef_action":
                vec = torch.zeros((n, dim), device=device, dtype=torch.float32)
            else:
                vec = torch.full(
                    (n, dim), float("nan"), device=device, dtype=torch.float32
                )
            per_robot[term_name] = vec
        actions[robot_name] = per_robot
    return actions


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="mobile_dual_reach_env"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: MobileDualReachEnv = gym.make(
        "MobileDualReachEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    while True:
        # cube_pose: [N, 2, 7] — index 0 = red (-> left hand), 1 = blue (-> right hand)
        cube_pose = obs["privilege_obs"]["cube_pose"]
        red_pose = cube_pose[:, 0, :]  # left hand target
        blue_pose = cube_pose[:, 1, :]  # right hand target

        action = _build_reach_action(env, red_pose=red_pose, blue_pose=blue_pose)
        step_result = env.step(action=action)
        obs = step_result[0]


if __name__ == "__main__":
    main()
