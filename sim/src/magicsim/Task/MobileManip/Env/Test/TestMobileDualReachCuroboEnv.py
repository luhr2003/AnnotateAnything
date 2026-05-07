"""Dual-arm reach smoke test using cuRobo IK action term (instead of pink IK).

Mirrors ``TestMobileDualReachEnv``:
  * dict-of-dict action layout per ``planner_manager.single_action_space``,
  * base = NaN (p-controller lock_skip → parked-base hold),
  * arm = 14-D ``[blue(7) | red(7)]`` driving R_ee / L_ee,
  * eef = zeros.

The only difference is the underlying robot yaml swaps
``arm_action: ik_pink`` for ``arm_action: ik_dual_curobo``. Pink IK is a
pure task-space tracker with no collision awareness — when the planned
trajectory or commanded pose grazes the desk the arm clips through it.
``DualCuroboIKActionCfg`` runs cuRobo's :class:`InverseKinematics` per
sim step with ``self_collision_check=True`` and the scene collision
world from ``planner_manager``, so the arm should refuse poses that
would collide.

Compare side-by-side with ``TestMobileDualReachEnv`` to see whether the
table-clipping behaviour is rooted in pink IK (problem disappears
here) or higher up the stack (problem persists).
"""

from magicsim.Task.MobileManip.Env.MobileDualReachEnv import MobileDualReachEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
import torch


# cuRobo dual-arm layout (matches pink IK): ``[right_pose(7) | left_pose(7)]``.
_POSE_DIM = 7
_DUAL_POSE_DIM = 2 * _POSE_DIM


def _term_dim(space) -> int:
    """Extract the feature dim from a gym Box / nested space."""
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"Cannot determine dim for space {space!r}")
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[-1])


def _build_reach_action(
    env: MobileDualReachEnv, red_pose: torch.Tensor, blue_pose: torch.Tensor
) -> dict:
    """Identical to the pink-IK test version — the cuRobo dual-arm
    action accepts the same 14-D ``[blue(7) | red(7)]`` payload
    (vega1psharpa.py:769 ``DualCuroboIKActionCfg.action_space`` matches
    ``ik_pink``'s 14-D layout)."""
    device = env.device
    n = red_pose.shape[0]
    actions: dict = {}

    planner_manager = env.scene.planner_manager
    for robot_name, robot_space in planner_manager.single_action_space.spaces.items():
        per_robot: dict = {}
        for term_name, term_space in robot_space.spaces.items():
            dim = _term_dim(term_space)
            if term_name == "base_action":
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
    version_base=None,
    config_path="../../Conf",
    config_name="mobile_dual_reach_curobo_env",
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
        red_pose = cube_pose[:, 0, :]
        blue_pose = cube_pose[:, 1, :]

        action = _build_reach_action(env, red_pose=red_pose, blue_pose=blue_pose)
        step_result = env.step(action=action)
        obs = step_result[0]


if __name__ == "__main__":
    main()
