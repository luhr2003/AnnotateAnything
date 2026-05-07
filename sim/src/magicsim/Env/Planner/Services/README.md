# Planner Services — IKServer / MotionGenServer Architecture

Authoritative reference for the four in-process planning services under
this directory. Covers request schemas, the request→slot→result mapping,
NaN-as-disable semantics, info_link / extra_fk_link wiring, paired-goalset
handling, and the post-flatten left/right (`hand_id`) contract.

**Scope**: live system state as of 2026-04. For historical design notes
see the cross-references in §10.

---

## 1. Module layout

```
src/magicsim/Env/Planner/Services/
  IKServer.py           (~1250 lines)  single-base InverseKinematics service
  DualIKServer.py       (~1207 lines)  locked+free-base InverseKinematics pair
  MotionGenServer.py    (~1100 lines)  single-base BatchMotionPlanner service
  DualMotionGenServer.py(~1180 lines)  locked+free-base BatchMotionPlanner pair
  batch_planning_utils.py              VariableBatchPlanner wrapper (pad-to-max_batch_size)
  nan_preprocessing.py                 detect_nan_disable_single / detect_nan_and_pad_goalset
  __init__.py                          _normalize_planner_devices helper
```

### 1.1 "Dual" means *locked+free base pair*, not *two arms*

The `Dual*` naming is orthogonal to arm count. It refers to the **solver
pair** on a mobile base:

- `lock_base=True`  → LOCKED solver. Targets transformed world→robot base
  frame. Uses the YAML whose filename has no `_mobile` suffix.
- `lock_base=False` → FREE solver. Targets stay in world frame. Uses the
  `_mobile` YAML that declares virtual base joints
  `(base_x, base_y, base_h, base_z)`.

`DualIKPlanRequest.lock_base` / `DualMotionGenPlanRequest.lock_base`
picks which solver inside the pair runs. Arm count (1 or 2 tool_frames)
is unrelated — both `DualIKServer` and `IKServer` handle any L. See §5
for the post-flatten left/right story.

### 1.2 Which service handles what

| Caller | Service | Goalset? | Returns |
|---|---|---|---|
| `AtomicSkill.Grasp._submit_goalset` | IKServer / DualIKServer | yes, G candidates | `(success, goalset_index, env_ids)` |
| `GlobalPlanner.MoveL._submit_ik` | IKServer / DualIKServer | no (G=1) | same |
| `GlobalPlanner.MoveL._submit_plan` | MotionGenServer / DualMotionGenServer | **no** | `(actions, success, env_ids)` |
| `GlobalPlanner.MobileServoL._submit_ik` | IKServer / DualIKServer | no (G=1) | same |
| `GlobalPlanner.*._submit_plan` | MotionGenServer / DualMotionGenServer | **no** | actions + success |

MotionGen does NOT support goalset — `MotionPlannerCfg.create(max_goalset=1)`
is hardcoded. It's single-pose only. Goalset candidate selection is an
IK-stage concern.

---

## 2. Request schemas

### 2.1 `IKPlanRequest` (IKServer.py)

```python
@dataclass
class IKPlanRequest:
    env_ids: List[int]                   # caller-space env ids
    target_pos: torch.Tensor             # 2-D or 3-D or 4-D — see below
    robot_states: Dict[str, torch.Tensor]   # {"base_pos", "base_quat", "joint_pos", "joint_vel"}
```

**Accepted `target_pos` shapes** (single + goalset unified; single == G=1):

| Shape | ndim | Meaning |
|---|---|---|
| `(N, L * 7)` | 2 | Single-pose. Reshaped to `(N, 1, L, 7)` internally. |
| `(N, L, 7)` | 3 (last=7) | Single-pose. Reshaped to `(N, 1, L, 7)`. |
| `(N, G, L * 7)` | 3 (last=L*7) | Per-frame goalset, flat. Reshaped to `(N, G, L, 7)`. |
| `(N, G, L, 7)` | 4 | Canonical goalset. Used directly. |

Where `L == len(tracked_tool_frames)`. Each 7-vec is
`[x, y, z, qw, qx, qy, qz]`. Position-only `(..., 3)` shapes are
**rejected** (assert fires).

**Shape disambiguation rule for `L=1`**: 3-D `(N, 1, 7)` and
`(N, G, 1*7) = (N, G, 7)` are syntactically identical. Both canonicalize
to `(N, G, L=1, 7)` — treat the middle dim as `G`.

**NaN-as-disable**: any xyz-NaN row disables that `(env, tool)` pair for
this solve only. See §7.

### 2.2 `DualIKPlanRequest` (DualIKServer.py)

Same schema + `lock_base: bool = True`. Selects locked vs free solver
inside the pair.

### 2.3 `MotionGenPlanRequest` (MotionGenServer.py)

```python
@dataclass
class MotionGenPlanRequest:
    env_ids: List[int]
    target_pos: torch.Tensor             # 2-D (N, eef_num*7) or 3-D (N, eef_num, 7)
    robot_states: Dict[str, torch.Tensor]
```

MotionGen is single-pose only. Accepted shapes:

| Shape | ndim | Meaning |
|---|---|---|
| `(N, eef_num * 7)` | 2 | Flat, reshaped to `(N, eef_num, 7)`. |
| `(N, eef_num, 7)` | 3 | Canonical. |

Position-only rejected. `eef_num == len(tracked_tool_frames)`.

### 2.4 `DualMotionGenPlanRequest`

Same + `lock_base: bool = False` (default free for mobile bases).

### 2.5 `robot_states` payload (all 4 requests)

```python
{
    "base_pos":   torch.Tensor,  # (num_envs, 3) — world-frame base position
    "base_quat":  torch.Tensor,  # (num_envs, 4) — world-frame base quat (w,x,y,z)
    "joint_pos":  torch.Tensor,  # (num_envs, num_sim_joints) — sim joint order
    "joint_vel":  torch.Tensor,  # (num_envs, num_sim_joints) — sim joint order
}
```

Tensors cover ALL envs (full sim batch). `_preprocess_request` slices
by `req.env_ids`.

---

## 3. Request → slot → result pipeline

This is the single most important invariant. Four index spaces coexist:

| Space | Range | Meaning |
|---|---|---|
| **game env_id** | arbitrary `int` (0..num_envs-1) | Caller's env id. `world_cfg_list` indexed by this. |
| **request-local `j`** | 0..`len(req.env_ids)-1` | j-th env in a single request. Caller reads results in this order. |
| **merged idx** | 0..`len(merged_env_ids)-1` | Position after dedup across all requests in one microbatch window. |
| **solver slot `b`** | 0..`max_batch_size-1` | Solver batch row. All solver APIs (`load_collision_model`, `update_tool_pose_criteria_per_env`, `solve_pose`, `result.success[b]`) use this. |

### 3.1 Pipeline (identical across all 4 services)

```
Caller              Request A: env_ids=[5,3,8], target_A, robot_states
                    Request B: env_ids=[3,12],  target_B, robot_states
                                     │
                        submit_ik / submit_plan
                                     ▼
Worker queue  ┌────────────────────────────────────────┐
 (per thread) │ wait microbatch_wait_ms, drain queue   │
              └────────────────────────────────────────┘
                                     │
                          _preprocess_request                 # per request
                                     │   slice robot_states[env_ids]
                                     │   canonicalize target_pos
                                     │   build active-DOF JointState
                                     │
              per-request tuples (env_ids, target, ...)
                                     │
                          Merge + dedup by env_id             # microbatch
                                     │
              merged_env_ids = [5, 3, 8, 12]   # unique, first-seen
              merged_targets = [tA[0], tA[1], tA[2], tB[1]]
              per_req_indices = {A:[0,1,2], B:[1,3]}          # j → merged idx
                                     │
                          Chunk by max_batch_size
                                     │
              batch_env_ids = merged_env_ids[s:e]             # e.g. [5,3,8,12]
              batch_target  = merged_target[s:e]
                                     ▼
                          _solve_one_batch / _plan_one_batch
                          ┌──────────────────────────────┐
                          │ for slot, env_id in          │
                          │         enumerate(bids):     │
                          │   scene = world_cfg_list     │   ◄── game-space read
                          │              [env_id]        │
                          │   load_collision_model(      │   ──► solver-space write
                          │     scene, env_idx=slot)     │
                          │                              │
                          │ detect_nan_and_pad_goalset   │
                          │                              │
                          │ for b in range(B_actual):    │
                          │   update_tool_pose_criteria  │   ◄── solver-space
                          │       _per_env(b, crit)      │
                          │                              │
                          │ goal = GoalToolPose.from...  │
                          │ result = solve_pose(         │
                          │   goal, current_state)       │
                          │                              │
                          │ extract success[b], gi[b]    │   ◄── solver-space
                          └──────────────────────────────┘
                                     │
              success_chunk[b] ↔ slot b ↔ batch_env_ids[b]
                                     │
                  merged_success[s:e] = success_chunk         # merge-idx write
                                     │
                          Per-request fan-out
                                     │
              success_out = [merged_success[ii]
                             for ii in per_req_indices[fut]]  # → request-local j
                                     ▼
                          fut.set_result((success_out, ..., env_ids))
```

### 3.2 Scene load is the ONLY game-env_id → slot crossover

```python
# Inside _solve_one_batch:
for slot, env_id in enumerate(batch_env_ids):     # slot = 0..B_actual-1
    scene = world_cfg_list[int(env_id)]           # ← GAME SPACE read
    load_collision_model(scene, env_idx=slot)     # ← SOLVER SLOT write
```

Everything else that happens inside `_solve_one_batch` (criteria,
target, current_state, result) uses slot `b` exclusively.

### 3.3 Dedup semantics

If env X appears in both request A (target T_A) and request B
(target T_B), merge keeps T_A (first seen). B's duplicate entry is
silently skipped. Both requests receive the SAME `success` / `goalset_index`
result for env X. This is intended: two clients asking about the same
env get consistent answers; the more recent target is dropped.

### 3.4 Pad slots `[B_actual..max_batch_size)`

- Scene: reloaded with `Scene()` on B-shrink only (pad-slot hygiene).
- Target: auto-padded by `VariableBatchPlanner` / `_pad_batch_inputs`
  to retract FK (trivially solvable).
- Criteria: stale — whatever was last written. Benign because the pad
  target is at FK (zero cost for either track or disabled).
- Result: sliced out before scatter.

---

## 4. YAML contract (curobo-side)

Each robot ships one (or a pair of) YAMLs under
`Third_Party/curobo/curobo/content/configs/robot/magicsim_*.yml`.

### 4.1 Fields read by PlannerManager

```yaml
robot_cfg:
  kinematics:
    tool_frames: [frame0, frame1, ...]     # TRACKED frames (drive IK/motiongen cost)
    base_link: "base_fixture_link"          # differs between locked/free pair
    lock_joints: {...}                      # optional, per-joint lock values
    # ... full curobo kinematics block ...

# Top-level (OUTSIDE robot_cfg):
extra_fk_link: ["base_link", "link2"]       # FK-only frames (ToolPoseCriteria.disabled())
info_links: ["base_link", "link2", "tool0", "tool1"]   # position-mode readout order
ignore_joints: {joint_name: fill_value}     # optional, see §4.3
add_joints: {joint_name: sim_index}         # optional, virtual base joint mapping
```

### 4.2 `tool_frames` + `extra_fk_link` merge

`PlannerManager._merge_extra_fk_link`:
1. Reads `robot_cfg.kinematics.tool_frames` (tracked).
2. Reads top-level `extra_fk_link` (FK-only).
3. Merges into a single `tool_frames` list on the `robot_cfg` it hands
   to cuRobo (dedup, tracked first, then extras).
4. Returns `(robot_cfg, extra_fk_link, info_links)` to the Server.

Server-side in `_IKInstance.__init__`:

```python
self._tracked_tool_frames = [f for f in merged_tool_frames
                             if f not in extra_fk_link]   # L == len(tracked)
# init-time broadcast criteria
for frame in merged_tool_frames:
    criteria[frame] = (
        ToolPoseCriteria.track_position_and_orientation(xyz, rpy)
        if frame in tracked_set
        else ToolPoseCriteria.disabled()
    )
ik_solver.update_tool_pose_criteria(criteria)
```

Extras contribute ZERO cost at IK-seed, main IK, and TrajOpt stages
(curobo propagates `update_tool_pose_criteria` to all three). They still
receive FK buffers so callers can read their pose post-solve via
`compute_kinematics`.

### 4.3 `info_links` — position-mode FK readout order

Only used by MotionGen when `mode: position`:

```python
# _trajectory_to_eef_pose returns:
#   shape (T, len(info_links) * 7)
# with poses stacked in info_links order.
```

If omitted in YAML, defaults to the original (pre-merge) tracked
`tool_frames` order. Must be a subset of merged `tool_frames`; validated
at init.

### 4.4 Locked + free pair consistency

For `dual_mode=True` robots, PlannerManager loads both
`magicsim_<type>.yml` (locked) and `magicsim_<type>_mobile.yml` (free)
and asserts **identical**:
- `extra_fk_link`
- `info_links`
- merged `kinematics.tool_frames`

The only field that legitimately differs is `kinematics.base_link`
(locked = arm root, free = mobile root). Any other divergence raises
`ValueError` at PlannerManager construction — no silent union.

### 4.5 `lock_joints` / `ignore_joints` / `add_joints`

- `lock_joints`: sim joints driven by a controller (not by the IK / plan
  output). Server skips these when building active-DOF `JointState`.
- `ignore_joints`: sim joints that exist but shouldn't influence IK. Dict
  maps name → constant fill value. Server skips AND supplies the
  constant in the FK readout path.
- `add_joints`: VIRTUAL joints (e.g. `base_x`, `base_y`, `base_h`,
  `base_z` for mobile bases). Dict maps name → `sim_state.joint_pos`
  index from which to read the value. Used only on free-base pairs.

Active DOF = `robot_dof_name` \ `lock_joints` \ `ignore_joints`, with
`add_joints` appended (inserted into `robot_dof_name` before filtering).

---

## 5. Left/right (`hand_id`) post-flatten

Before the `MERGE_LEFT_RIGHT.md §1–§8` flatten, each dual-arm robot
spawned TWO servers (`_right.yml` + `_left.yml`) and AtomicSkills /
GlobalPlanners routed by `hand_id`. After flatten:

### 5.1 One server per robot

```python
PlannerManager.ik_server:        Dict[str, Union[IKServer, DualIKServer]]
PlannerManager.motiongen_server: Dict[str, Union[MotionGenServer, DualMotionGenServer]]
```

Keyed by `robot_name` only. No more inner `[hand_id]` dict. PlannerManager
rejects legacy per-arm YAML (raises if `planner.ik` is missing `type:`
at top level).

### 5.2 `hand_id` in action headers = target-packing signal

`hand_id` still rides in every GlobalPlanner action header
`((robot_id, hand_id, planner_mode), target_tensor)`, but now only
drives caller-side target packing:

| `hand_id` | Caller builds | Effect after NaN-disable |
|---|---|---|
| `0` (right) | 7-vec, right pose + NaN row for left | Solver disables left for this env |
| `1` (left) | NaN row for right + 7-vec, left pose | Solver disables right for this env |
| `-1` (both) | 14-vec, both poses | Both arms driven |

The caller's `_format_target_for_submit` packs the NaN rows; the Server's
`detect_nan_disable_single` / `detect_nan_and_pad_goalset` picks them up
and flips the per-env per-tool criterion to `disabled()`. See §7.

### 5.3 Unified curobo YAMLs

```
magicsim_dual_piper.yml          tool_frames: [R_link6, L_link6]
magicsim_dual_arx_x5.yml         tool_frames: [R_link6, L_link6]
magicsim_dual_so101.yml          tool_frames: [R_Fixed_Jaw, L_Fixed_Jaw]
magicsim_xtrainer.yml            tool_frames: [R_ee, L_ee]
magicsim_g1_simple.yml           tool_frames: [right_ee, left_ee]
magicsim_mobile_x7s.yml          tool_frames: [link20_tip, link11_tip]
magicsim_vega1p_sharpa.yml       tool_frames: [R_ee, L_ee]
```

Convention: **right first, left second**. Tracked tool_frames slot 0
= right arm, slot 1 = left arm. AtomicSkill code never reads this list
— it packs targets by its own `hand_id` convention and lets the NaN
row tell the server which arm is disabled.

---

## 6. Paired goalset (IK only)

### 6.1 What "paired" means

With `L >= 2` tracked frames and `G >= 2` goalset candidates, the IK
kernel can argmin in two modes:

- **unpaired** (default false): each frame independently picks
  `g_l = argmin_g cost(current_l, goal_l[g])`. Different frames can
  pick different `g`s.
- **paired** (new): all frames share a single `g` per (env, horizon
  step):
  ```
  g* = argmin_g sum_l cost(current_l, goal_l[g])
  for l: g_l = g*
  ```
  Needed for bimanual rigid grasps where slot `g` on right and slot `g`
  on left are jointly reachable.

`L=1` ⇒ paired silently reduces to unpaired (kernel dimension collapses).
`G=1` ⇒ paired == unpaired trivially.

### 6.2 YAML opt-in

Default `true` for IK. Set `paired: false` only for robots that
explicitly want independent argmins.

```yaml
planner:
  ik:
    type: "dual_piper"
    enable: true
    # ... other fields ...
    paired: true              # default; written explicitly for clarity
  motiongen:
    type: "dual_piper"
    enable: true
    # NO paired field — motion planning is single-pose
```

### 6.3 Plumbing

```
YAML planner.ik.paired
      │
      ▼
PlannerManager._create_ik_server:
    paired = bool(ik_config.get("paired", True))
      │
      ▼
IKServer(..., paired=paired)
      │
      ▼
_IKInstance.__init__:
    if self._paired:
        enable_paired_tool_pose(ik_cfg.core_cfg)   # BEFORE InverseKinematics(cfg)
      │
      ▼
Solver build:
    tool_pose_cfg.paired = True
    → paired warp kernel + ToolPoseDistancePerEnvPaired autograd
```

`enable_paired_tool_pose` lives at
`Third_Party/curobo/curobo/_src/solver/solver_core_cfg.py`.

MotionGen servers do NOT accept or plumb `paired` (single-pose only).

---

## 7. NaN-as-disable (runtime, per-env per-tool)

### 7.1 Caller convention

A 7-vec row is "NaN-as-disable" when the xyz component
(`[..., :3].isnan().any(dim=-1)`) is all-NaN. The Server interprets it
per `(env, tool)`:

- **All** candidates NaN → disable that tool for that env this solve.
- **Some** NaN (goalset only) → mod-pad NaN slots from the tool's real
  (non-NaN) candidates (`g_pad → real[g % real_count]`).
- **None** NaN → track normally.

### 7.2 `paired=True` equal-count rule

Inside one env, if ≥2 active tools (non-fully-NaN) exist, their
`real_count` MUST be equal. Otherwise `detect_nan_and_pad_goalset` raises
`ValueError`. Paired argmin requires jointly reachable candidate `g`.

`paired=False` is unconstrained — different tools can have different
real counts; shorter ones mod-pad.

### 7.3 Runtime, not init

Every `_solve_one_batch` / `_plan_one_batch` rewrites per-env criteria
rows for every (slot, tracked_frame) in the chunk:

```python
for b in range(B_actual):
    crit = {frame: (disabled_template if fully_nan[b, li]
                    else track_template)
            for li, frame in enumerate(tracked)}
    solver.update_tool_pose_criteria_per_env(b, crit)  # SLOT b, not env_id
```

No state carries between solves. env 5 can drive both arms in solve N
and only the right arm in solve N+1, with no stale criteria bleeding.

### 7.4 NaN value substitution

After marking disabled / padded, the Server substitutes the NaN slots
with current FK (via `compute_kinematics(current_state)`) so cuRobo
never sees NaN goals. The substituted value is inert: for disabled tools
the criterion weight is zero; for padded slots the target equals a real
candidate.

### 7.5 Helpers

```python
# nan_preprocessing.py
def detect_nan_disable_single(target: torch.Tensor) -> torch.Tensor:
    """(B, L, 7) → (B, L) bool. True means this (env, tool) is NaN-masked."""

def detect_nan_and_pad_goalset(
    target: torch.Tensor,         # (B, L, G, 7)
    tracked_frames: List[str],    # len L, used only for error messages
    paired: bool,
) -> Tuple[
    torch.Tensor,    # (B, L, G, 7) padded (NaN slots → mod-cycled real candidates)
    torch.Tensor,    # (B, L) bool — fully-NaN tools (caller substitutes FK)
    torch.Tensor,    # (B, L) int — real_count per (env, tool)
]:
    """Raises on paired unequal active-real_counts."""
```

---

## 8. FK / info-only frame handling

### 8.1 Non-tracked frames ≡ `extra_fk_link`

Frames declared in top-level `extra_fk_link` are:
- Merged into cuRobo's `kinematics.tool_frames` (so FK buffers are
  allocated).
- Marked `ToolPoseCriteria.disabled()` at init (zero cost everywhere).
- Filled with current-FK pose inside `_solve_one_batch` / `_plan_one_batch`
  before `GoalToolPose.from_poses` (cuRobo asserts no-NaN).

### 8.2 `compute_kinematics` called ONCE per batch

Previously there were two FK calls (one for info-only filler, one for
NaN substitution). Unified into one call; the result dict is reused for
both purposes. See e.g. `IKServer.py:_solve_one_batch`.

### 8.3 `mode: position` trajectory readout (MotionGen only)

`_trajectory_to_eef_pose(action_traj, base_pos, base_quat, joint_pos,
action_joint_names)` returns `(T, len(info_links) * 7)` with poses stacked
in `info_links` order. World frame when `relative_to_world_frame=True`,
robot-base frame otherwise. Used by `GlobalPlanner._advance_trajectory`
to return per-step pose targets instead of joint targets.

---

## 9. Pool + microbatch architecture

### 9.1 Instance pool

Each Server holds a list of `_IKInstance` / `_MotionGenInstance` worker
threads (configurable via `ik_num_instances` / `motiongen_num_instances`
in YAML). Requests are load-balanced: idle-first, else least-loaded
queue.

### 9.2 Microbatch window

Each instance's worker waits up to `microbatch_wait_ms` for more
requests before solving, so N parallel callers (one per env) coalesce
into one batched solve.

### 9.3 Group keys

Within a microbatch window, requests are grouped by:

- **IKServer** / **DualIKServer**: `G` (num_goalset). Different G values
  can't share a solve because the warp kernel JITs per `num_goalset`.
- **DualIKServer** / **DualMotionGenServer**: `lock_base` (picks which
  solver in the pair). Then by G for IK.
- **MotionGenServer**: no grouping (always G=1).

### 9.4 Pad-slot hygiene + VariableBatchPlanner

IK uses direct `solve_pose` with auto-pad; Servers track `_last_B` and
reload `Scene()` into pad slots only when B shrinks.

MotionGen uses `VariableBatchPlanner` (`batch_planning_utils.py`) which
pads current_state + goal + pad-slot scenes to `max_batch_size` with
retract-FK dummy problems and slices the result back. Overhead ~4%.

### 9.5 Instance ctor signature (accepted knobs)

```python
IKServer(
    robot_manager,                    # for joint-name lookup
    robot_cfg,                        # merged (tracked + extras) tool_frames
    robot_name,
    device,                           # sim-side device (not planner)
    batch_size,                       # = max_batch_size on solver
    microbatch_wait_ms,
    num_instances,                    # worker pool size
    num_seeds,                        # IK seeds per solve
    position_threshold, rotation_threshold,
    max_goalset,                      # largest G this Server will ever see
    extra_fk_link, info_links,        # §4
    track_xyz_weight, track_rpy_weight,
    debug,
    relative_to_world_frame,          # single-base only
    planner_devices,                  # per-instance GPU (multi-GPU)
    paired,                           # §6
)

DualIKServer(                         # same as IKServer + dual-specific:
    robot_cfg_locked, robot_cfg_free,
    base_joint_names,                 # virtual base joints
    robot_add_joints, robot_ignore_joints, robot_lock_joints,
    robot_dof_name_active,
    num_seeds_locked, num_seeds_free,
    # NO relative_to_world_frame (lock_base picks per-request)
)

MotionGenServer(                      # single-pose, no goalset / paired
    robot_manager, robot_cfg, robot_name,
    robot_dof_name, robot_dof_name_active,
    robot_lock_joints, robot_ignore_joints, robot_add_joints,
    device,
    batch_size, microbatch_wait_ms, num_instances,
    debug,
    mode,                             # "joint" or "position"
    info_links, extra_fk_link,
    track_xyz_weight, track_rpy_weight,
    max_attempts, enable_graph_attempt,
    relative_to_world_frame,
    planner_devices,
)

DualMotionGenServer(                  # MotionGenServer + dual-specific
    robot_cfg_locked, robot_cfg_free,
    base_joint_names,
    # NO relative_to_world_frame
)
```

---

## 10. Cross-references

This file consolidates the authoritative spec; these historical docs
still live in the repo root for context:

- `MERGE_LEFT_RIGHT.md` — §1–§8 left/right flatten plan (complete);
  §9 NaN-disable + per-frame goalset (shipped, this document).
- `ServiceMigrate.md` — v1→v2 Service port (info_links / extra_fk_link
  design, `ToolPoseCriteria.disabled()` plumbing rationale).
- `CUROBO_V2_02_MIGRATION_PLAN.md` — curobo API deltas
  (`MotionGen.plan_batch_env` → `BatchMotionPlanner.plan_pose`,
  `IKSolver.solve_batch_env` → `InverseKinematics.solve_pose`, etc.).
- `CUROBO_V2_03_DYNAMIC_BATCH.md` — `VariableBatchPlanner` pad-slot
  design.
- `PER_ENV_TOOL_POSE_COST_PLAN.md` — per-env weight buffer spec
  (`update_tool_pose_criteria_per_env` hook).
- `GOALSET_PER_FRAME_ANALYSIS.md` — `(B, L, G, 7)` per-frame goalset
  design notes.
- `extra_info_links.md` — YAML `extra_fk_link` / `info_links` contract.
- `CUROBO_V2_01_CURRENT_INTERFACES.md` — pre-migration v1 reference.
- `CUROBO_V2_04_ENV_MIGRATION.md` — per-env world cfg + scene cache.

**Curobo-side feature dependencies** (not MagicSim):
- `Third_Party/curobo/curobo/_src/solver/solver_core_cfg.py::enable_paired_tool_pose`
- `Third_Party/curobo/curobo/_src/cost/wp_tool_pose.py::ToolPoseDistancePerEnvPaired`
- `Third_Party/curobo/curobo/_src/cost/tool_pose_criteria.py::ToolPoseCriteria.{track_position_and_orientation, disabled}`
- `Third_Party/curobo/curobo/inverse_kinematics.py::InverseKinematics.{update_tool_pose_criteria, update_tool_pose_criteria_per_env}`

---

## 11. Invariants at a glance (things that MUST hold)

1. **Scene load is the ONLY game-env_id → solver-slot crossover.** Everywhere
   else, `b` is the batch slot, not the caller's env id.
2. **`update_tool_pose_criteria_per_env(b, ...)`** is indexed by slot `b`,
   matching `load_collision_model(env_idx=b)`. Never pass `batch_env_ids[b]`
   here.
3. **Every solve rewrites every (slot, tracked_frame) criteria row** for
   real slots in the chunk. No state carries across solves.
4. **`target_pos` never contains NaN after preprocessing.** NaN rows are
   substituted with FK before `GoalToolPose.from_poses`.
5. **Locked + free YAMLs of a dual-mode robot must declare identical
   `extra_fk_link`, `info_links`, `tool_frames`.** PlannerManager asserts
   this at init.
6. **`paired=True` + goalset + unequal active-tool real_counts ⇒ raise.**
   Paired argmin requires jointly reachable `g`.
7. **Dedup is by game env_id within one microbatch window.** Duplicate
   env gets the first-seen target's result; other requests referencing
   the same env get the SAME answer.
8. **Pad slots `[B_actual..max_batch_size)` are benign.** Their scene is
   `Scene()` (empty) on B-shrink, their target is retract-FK, their
   criteria are stale but irrelevant, their results are sliced out.
9. **`hand_id` does NOT select a server.** Post-flatten it only drives
   caller-side target packing (NaN rows for inactive arms).
10. **MotionGen is single-pose.** `max_goalset=1` hardcoded; no `paired`
    flag; no goalset fields in the request.
