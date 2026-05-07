# Bimanual_dexhand Runbook

## Purpose

This project builds single-hand and bimanual dexterous grasp candidates in Isaac Sim from:

- an object USD
- a hand package under `assets/hands/<hand_name>/`
- global category and optimizer configs under `configs/`

Current categories:

- `cat1`: edge / rim grasp
- `cat2`: lower-edge support / hold
- `cat3`: bimanual side hold with opposite-side anchors at similar height
- `cat4`: convex-hold from direct USD input, with side and top-down anchor families

All Isaac entrypoints in this repo should be run with:

```bash
$ISAACSIM_PYTHON ...
```

Do not use plain `python3` for the runtime scripts, because they import `isaacsim`.

---

## Required Files For A New Hand

For a new hand family `<hand_name>`, the runtime-required layout is:

```text
Bimanual_dexhand/
  assets/
    hands/
      <hand_name>/
        asset/
          <left_hand>.usd
          <right_hand>.usd
          <left_hand>.urdf
          <right_hand>.urdf
          meshes/...
        collision/
          left_collision.yaml
          right_collision.yaml
        config/
          left_hand.yaml
          right_hand.yaml
          contact_points.yaml
          pose_seeds.yaml
```

Global files that must also exist:

- `configs/category_config.yaml`
- `configs/optimizer.yaml`

Files that may exist in a hand package but are not required by the main runtime loader today:

- `config/bimanual.yaml`
- `config/bimanual_rule.yaml`
- `config/optimizer.yaml`

The loader path is in [config_loader.py](Bimanual_dexhand/src/config_loader.py). The actual required files are loaded in `load_all_for_hand(...)`.

---

## What Each Hand File Must Define

### `config/left_hand.yaml` and `config/right_hand.yaml`

These define the hand structure:

- asset paths
- wrist link and palm link
- hand frame convention
- controllable joints
- joint limits
- default postures
- link tips and collision candidate links

Important fields:

- `asset.usd_path`
- `asset.urdf_path`
- `root.wrist_link`
- `root.palm_link`
- `frame_convention.palm_normal_local`
- `frame_convention.finger_forward_local`
- `frame_convention.thumb_opposition_local`
- `joints.controllable`
- `joints.groups`
- `default_postures`

### `collision/left_collision.yaml` and `collision/right_collision.yaml`

These define the collision proxy model:

- default joint positions for collision interpretation
- self-collision ignore pairs
- self-collision buffers
- collision spheres grouped by link

Important fields:

- `default_joint_positions`
- `self_collision.ignore`
- `self_collision.buffer`
- `geometry.collision_spheres.spheres`

### `config/contact_points.yaml`

This defines semantic points attached to collision spheres, plus category usage.

Important fields:

- `left_points`
- `right_points`
- `cat1.active_points`
- `cat1.avoid_points`
- `cat1.opposition_pairs`
- `cat2.active_points`
- `cat2.avoid_points`
- `cat2.opposition_pairs`
- `cat3.active_points`
- `cat3.avoid_points`
- `cat3.opposition_pairs`

Each semantic point must map to:

- `source_link`
- `source_sphere_index`
- `role_tags`

### `config/pose_seeds.yaml`

This defines rough seed generation:

- which posture presets are available
- category seed templates
- seed posture names
- approach families
- wrist perturbation ranges
- left/right seed adjustments

Important fields:

- `shared_posture_names.available`
- `left_seed_adjustment`
- `right_seed_adjustment`
- `cat1`
- `cat2`
- `cat3`

---

## Hand Semantics We Assume

The code is now more tolerant of different finger counts for `cat1`, `cat2`, and `cat3`, but there is still one semantic assumption:

- one thumb
- any number of non-thumb fingers among `index`, `middle`, `ring`, `little`

That means:

- different finger counts are okay
- finger directions and palm / thumb directions can be defined from YAML
- `cat1`, `cat2`, and `cat3` will operate over all configured non-thumb fingers
- completely different digit naming or a non-thumb-less topology still needs code changes

For a new hand, make sure the semantic points and joint groups still follow the thumb / finger naming convention the runtime expects.

---

## Common Paths

In the commands below, we usually set:

```bash
HAND_DIR=Bimanual_dexhand/assets/hands/dex3_1
CONFIG_DIR=Bimanual_dexhand/configs
OBJECT_USD=Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd
```

You can replace those with another hand package or object.

---

## Command Runbook

### 1. Build or refresh the object SDF

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_build_object_sdf.py \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --voxel_size 0.005 \
  --padding_voxels 8 \
  --headless
```

Use this when you want the SDF cached before staged optimization.

### 2. Visualize anchor proposals for one hand

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --side right \
  --category cat2 \
  --top_k 20 \
  --visualize
```

Useful for checking whether anchor selection is correct before any seed or optimization work.

### 3. Visualize one-hand seed poses

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_seed_preview.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --side right \
  --category cat2 \
  --anchor_rank 0 \
  --num_seeds_per_contact 5 \
  --seed_rank 0
```

Change:

- `--side left` to inspect the left hand
- `--anchor_rank` to inspect another anchor
- `--seed_rank` to inspect another seed

### 4. Visualize one-hand staged poses

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_staged_pose_preview.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --side right \
  --category cat2 \
  --top_k_anchors 1 \
  --num_seeds_per_contact 5 \
  --top_k_seeds_per_contact 1 \
  --bundle_rank 0 \
  --result_rank 0 \
  --stage squeeze
```

Use this for the optimized `pregrasp`, `grasp`, or `squeeze` pose.

Note:

- `run_single_hand_staged_pose_preview.py` supports `cat1`, `cat2`, `cat3`, and `cat4`
- `run_single_hand_staged.py` is the headless JSON-export runner for `cat1` through `cat4`

### 4b. Visualize cat4 convex-hold scaffold

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_staged_pose_preview.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Object.usd \
  --side right \
  --category cat4 \
  --top_k_anchors 3 \
  --num_seeds_per_contact 8 \
  --top_k_seeds_per_contact 1 \
  --bundle_rank 0 \
  --result_rank 0 \
  --stage squeeze
```

`cat4` is currently intended for single-hand convex-like holds. It uses the
direct USD mesh, not convex piece assets. It now mixes:

- side-band `cat4` anchors for bottle / can style holds
- top-cap `cat4` anchors for smaller convex objects where top-down grasps are useful

Top anchors are tagged with `cat4_grasp_mode: "top"` in the JSON and use the
`top_down` seed approach.

### 4c. Headless cat4 JSON export for large batches

For high-volume `cat4` generation, use the headless staged runner rather than
the preview script. This saves the full staged summary as JSON.

If you want a file that
[single_hand_physics_validation.py](Bimanual_dexhand/single_hand_physics_validation.py)
can consume directly, use `--save_format validation` or `--save_format both`.
That writes the standardized single-hand validation schema:

- top-level `type`
- top-level `bottom_center`
- top-level `functional_grasp.body`
- per-entry `coarse_grasp` / `fine_grasp` / `final_grasp`
- per-stage `position` / `orientation` / `joints`

Example: about `500` optimized `cat4` poses

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_staged.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/apple/apple_001/Object.usd \
  --side right \
  --category cat4 \
  --num_surface_points 20000 \
  --top_k_anchors 25 \
  --num_seeds_per_contact 20 \
  --top_k_seeds_per_contact 20 \
  --top_k_optimized_per_contact 20 \
  --max_workers 16 \
  --headless
```

That configuration yields up to:

- `25` anchor bundles
- `20` optimized seeds per bundle
- up to `500` optimized staged pose entries in the saved JSON

For about `1000` poses, double the anchor count:

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_staged.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/apple/apple_001/Object.usd \
  --side right \
  --category cat4 \
  --num_surface_points 20000 \
  --top_k_anchors 50 \
  --num_seeds_per_contact 20 \
  --top_k_seeds_per_contact 20 \
  --top_k_optimized_per_contact 20 \
  --max_workers 16 \
  --headless
```

Notes:

- `--top_k_seeds_per_contact` controls how many seeds are actually optimized.
- `--top_k_optimized_per_contact` controls how many optimized results per anchor bundle are written to JSON.
- use `0` for `--top_k_optimized_per_contact` if you want to dump every optimized result in each bundle.
- for `cat4`, `20000` surface samples is the intended practical default range; `100000` is usually unnecessary.
- if you omit `--save_json` in headless mode, the staged runner now saves automatically under `outputs/single_hand_staged/<category>/`.
- `--save_format summary` keeps the staged-debug summary format.
- `--save_format validation` writes a validator-ready JSON under `outputs/physics_validation_inputs/<category>/`.
- `--save_format both` writes both files.

Example: validator-ready `cat4` apple batch with about `100` poses

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_staged.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/apple/apple_001/Object.usd \
  --side right \
  --category cat4 \
  --num_surface_points 20000 \
  --top_k_anchors 25 \
  --num_seeds_per_contact 7 \
  --top_k_seeds_per_contact 4 \
  --top_k_optimized_per_contact 4 \
  --max_workers 8 \
  --headless \
  --save_format validation \
  --save_json Bimanual_dexhand/outputs/physics_validation_inputs/cat4/cat4_right_apple_100_grasps.json
```

### 5. Run bimanual staged generation

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --primary_side right \
  --category cat2 \
  --top_k_anchors 3 \
  --num_opposite_candidates 3 \
  --top_k_anchor_pairs 3 \
  --num_seeds_per_contact 3 \
  --top_k_seeds_per_contact 1 \
  --max_workers 8
```

Notes:

- `--max_workers 0` means automatic worker count
- `--pose_source staged` is the default
- `--stage squeeze` is the default staged pose shown in the scene

### 5b. Cat3 side-hold behavior

`cat3` is now intended for bimanual side holding of bulky side-graspable objects:

- anchors come from the side band of the object
- left/right anchor pairing prefers opposite sides at similar `z` height
- the hand is oriented for a vertical side hold:
  - fingers start downward, but cat3 optimization may roll or raise the hand to keep the full hand above the object bottom
  - palm facing inward toward the object
  - thumb used as side support rather than forcing palm contact

This is important for hands like `dex3_1`, where the thumb geometry can block
true palm-on-object contact during a side hold.

A good first headless cat3 validation export is:

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Object.usd \
  --primary_side right \
  --category cat3 \
  --num_surface_points 30000 \
  --top_k_anchors 24 \
  --num_opposite_candidates 6 \
  --top_k_anchor_pairs 120 \
  --num_seeds_per_contact 12 \
  --top_k_seeds_per_contact 8 \
  --top_k_optimized_per_side 4 \
  --max_result_pair_checks 16 \
  --max_workers 8 \
  --parallel_backend process \
  --headless \
  --save_format validation \
  --save_json Bimanual_dexhand/outputs/physics_validation_inputs/cat3_bimanual_200.json
```

For Sharpa cat3 on `Bimanual_dexhand/assets/objects/Object.usd`,
use the fast contact-then-squeeze path. Cat3 squeeze is exported as joint-only:
the wrist is placed with the palm-first contact posture, then the final grasp
keeps that grasp wrist pose and applies the squeeze joints. For Sharpa cat3,
those squeeze joints use the mild `cat3_side_press` posture rather than a
cat1-style finger curl.
Cat3 pair generation uses parallel side bands: for each side anchor it keeps the
closest few anchors on the opposite side instead of using a direct farthest-point
match through the object. It rejects cross-diagonal pairs by requiring the
opposite anchors to stay in the same local cross-section, and the palm direction
is aligned to the side-normal push axis rather than the raw point-to-point
diagonal. Cat3 pair selection also ranks cross-section closeness before raw
object width, even when using the farthest-pair selector.
Cat3 optimization also enforces a bottom-clearance check against the object
bbox: candidate poses are allowed to move upward and roll about the palm normal,
but the palm contact rank remains tight before finger-only squeeze is exported.
The exported cat3 `coarse_grasp` is rebuilt from the selected safe contact pose
as a side-start pose: same wrist orientation and height as contact, retreated
outward along the horizontal side normal. It is checked too, so `coarse_grasp`,
`fine_grasp`, and `final_grasp` all keep the hand collision spheres above the
object bottom without creating a top-down approach arc.

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/sharpa \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Object.usd \
  --category cat3 \
  --primary_side right \
  --headless \
  --save_json Bimanual_dexhand/outputs/physics_validation_inputs/sharpa_cat3_object_200.json \
  --save_format validation \
  --top_k_anchors 20 \
  --num_opposite_candidates 5 \
  --top_k_anchor_pairs 120 \
  --top_k_seeds_per_contact 1 \
  --top_k_optimized_per_side 1 \
  --max_result_pair_checks 1 \
  --max_workers 16 \
  --parallel_backend auto \
  --pair_selector farthest_filtered \
  --settle_steps 1
```

To visualize one final Sharpa cat3 bimanual pose in Isaac Sim, omit
`--headless` and keep the stage as `squeeze`. To inspect only the hand-motion
contact pose before finger closing, change `--stage squeeze` to `--stage grasp`.

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/sharpa \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Object.usd \
  --category cat3 \
  --primary_side right \
  --stage squeeze \
  --pose_source staged \
  --pair_selector farthest_filtered \
  --top_k_anchors 12 \
  --num_opposite_candidates 5 \
  --top_k_anchor_pairs 12 \
  --top_k_seeds_per_contact 1 \
  --top_k_optimized_per_side 1 \
  --max_result_pair_checks 1 \
  --max_workers 8 \
  --parallel_backend thread \
  --settle_steps 1
```

### 6. Visualize bimanual seed poses instead of staged poses

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --primary_side right \
  --category cat2 \
  --top_k_anchors 3 \
  --num_opposite_candidates 3 \
  --top_k_anchor_pairs 3 \
  --num_seeds_per_contact 5 \
  --pose_source seed \
  --seed_rank 0 \
  --max_workers 8
```

This is the best command when we want to debug left/right hand orientation before optimization.

### 7. Show raw pair overlays for debugging

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --primary_side right \
  --category cat2 \
  --top_k_anchors 3 \
  --num_opposite_candidates 3 \
  --top_k_anchor_pairs 3 \
  --num_seeds_per_contact 3 \
  --top_k_seeds_per_contact 1 \
  --show_debug_overlays \
  --pair_visualization all \
  --max_workers 8
```

### 8. Headless save of bimanual results

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Chair/chair_001/Object.usd \
  --primary_side right \
  --category cat2 \
  --top_k_anchors 3 \
  --num_opposite_candidates 3 \
  --top_k_anchor_pairs 3 \
  --num_seeds_per_contact 3 \
  --top_k_seeds_per_contact 1 \
  --max_workers 8 \
  --headless \
  --save_json outputs/bimanual_cat2_test.json \
  --save_format validation
```

Notes:

- `--save_format summary` keeps the preview/debug summary JSON.
- `--save_format validation` writes the legacy physics-validation schema:
  - top-level `functional_grasp.body`
  - per-pair `left_hand` / `right_hand`
  - per-hand `coarse_grasp` / `fine_grasp` / `final_grasp`
  - stage fields `position`, `orientation`, `joints`
- For headless bimanual validation exports with `--save_json`, `validation`
  writes only the validator-ready JSON file and does not dump the preview
  summary JSON to stdout.
- `--save_format both` writes both the summary JSON and a sibling `*_validation.json`.

---

## Physics Validation

The current [physics_validation.py](Bimanual_dexhand/physics_validation.py) is not a real CLI script yet. It is driven by top-of-file constants.

Before running it, edit these constants inside the file:

- `HAND_TYPE`
- `HAND_RUNTIME_MODE`
- `HAND_USD_OVERRIDES`
- `OBJECT_USD_PATH`
- `GRASP_JSON_PATH`
- `NUM_ENVS`
- optionally `INVERSE_GRAVITY`

Important:

- keep the hand assets instanceable for parallel cloning
- for property changes, only change the scalar constants you need
- do not use this script to rewrite collider settings unless we explicitly decide to do that

If you want to validate with the newer `dex3_1` hand USDs that already contain
their updated collision model, keep:

- `HAND_TYPE = "dex3_1"`

and switch only the validation-time runtime path:

- `HAND_RUNTIME_MODE = "authored_usd"`
- `HAND_USD_OVERRIDES["left"] = "/abs/path/to/new_left_dex3_1.usd"`
- `HAND_USD_OVERRIDES["right"] = "/abs/path/to/new_right_dex3_1.usd"`

In that mode the validator:

- still uses the `assets/hands/dex3_1/config/*.yaml` files for joint order, palm link, and approach direction
- loads the override USDs for simulation
- preserves the authored collision geometry from those newer hand USDs
- configures the authored hand joints directly instead of running the older generic hand-collision rewrite path

The runtime command is then:

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/physics_validation.py
```

For bimanual `cat1` / `cat2` / `cat3`, the intended input is now the legacy-style
JSON with:

- top-level `functional_grasp.body`
- `left_hand` / `right_hand`
- `coarse_grasp` / `fine_grasp` / `final_grasp`
- `position`, `orientation`, `joints`

The validator also still accepts the newer top-level `grasps` format for
compatibility.

The preferred export path for bimanual validation is:

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_bimanual.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Object.usd \
  --primary_side right \
  --category cat1 \
  --top_k_anchors 25 \
  --num_opposite_candidates 5 \
  --top_k_anchor_pairs 120 \
  --num_seeds_per_contact 20 \
  --top_k_seeds_per_contact 20 \
  --top_k_optimized_per_side 4 \
  --max_workers 16 \
  --headless \
  --save_json Bimanual_dexhand/outputs/physics_validation_inputs/cat1_bimanual.json \
  --save_format validation
```

For validation exports, `run_bimanual.py` now defaults to
`--export_speed_profile safe`. That profile keeps the same pipeline structure but
reduces cold-start cost by:

- lowering staged solver iteration budgets
- reducing expensive stage-init roll/pitch sweep counts
- capping final contact-seek retries and stopping once a successful refinement is found

If you want the original slower behavior for comparison, add:

```bash
--export_speed_profile standard
```

For headless validation exports, `run_bimanual.py` also defaults to
`--parallel_backend auto`, which prefers process-based side-bundle workers and
falls back to threads if the environment does not support that cleanly. You can
force the backend with:

```bash
--parallel_backend process
```

or

```bash
--parallel_backend thread
```

For faster bimanual export, the expensive part is staged optimization, not just
pair counting. `run_bimanual.py` now separates:

- `--top_k_seeds_per_contact`: how many seeds we generate and rank per side bundle
- `--top_k_optimized_per_side`: how many of those ranked seeds we actually run
  through staged optimization

If `--top_k_optimized_per_side` is omitted, the runner defaults to
`min(top_k_seeds_per_contact, 4)`, which keeps the high-level ranking behavior
close while cutting most of the cost.

For `cat3`, the pairing logic now also prefers opposite-side anchors with
similar `z` height, so it is usually better to keep enough anchor breadth
(`--top_k_anchors`, `--num_opposite_candidates`, `--top_k_anchor_pairs`) before
raising per-side optimization depth.

The headless single-hand staged runner can now emit a single-hand legacy-style
validation JSON directly:

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/run_single_hand_staged.py \
  --hand_dir Bimanual_dexhand/assets/hands/dex3_1 \
  --config_dir Bimanual_dexhand/configs \
  --object_usd Bimanual_dexhand/assets/objects/Object.usd \
  --side right \
  --category cat1 \
  --top_k_anchors 25 \
  --num_seeds_per_contact 20 \
  --top_k_seeds_per_contact 20 \
  --top_k_optimized_per_contact 20 \
  --max_workers 16 \
  --headless \
  --save_format validation
```

That output uses:

- top-level `type`
- top-level `bottom_center`
- top-level `functional_grasp.body`
- per-entry `coarse_grasp` / `fine_grasp` / `final_grasp`
- stage fields `position`, `orientation`, `joints`

The `joints` array follows the selected hand's configured controllable-joint
order, so different hand families can emit different joint counts and orders
without changing the export code.

There is also a dedicated single-hand validator now:

```bash
$ISAACSIM_PYTHON Bimanual_dexhand/single_hand_physics_validation.py
```

Unlike the bimanual validator, it:

- reads one-hand entries from `functional_grasp.body`
- uses only one hand instance
- keeps the object floating initially with gravity disabled
- turns gravity on only during the hold / lift validation phase

Before running it, edit the top-of-file constants in
[single_hand_physics_validation.py](Bimanual_dexhand/single_hand_physics_validation.py):

- `HAND_TYPE`
- `HAND_RUNTIME_MODE`
- `HAND_USD_OVERRIDES`
- `HAND_SIDE`
- `OBJECT_USD_PATH`
- `GRASP_JSON_PATH`
- `NUM_ENVS`

For the `cat4` apple example above, set:

- `OBJECT_USD_PATH = "Bimanual_dexhand/assets/objects/apple/apple_001/Object.usd"`
- `GRASP_JSON_PATH = "Bimanual_dexhand/outputs/physics_validation_inputs/cat4/cat4_right_apple_100_grasps.json"`

The same newer-hand switch is available here too. For example, if the right-hand
validator should use a newer authored-collision `dex3_1` USD:

- `HAND_TYPE = "dex3_1"`
- `HAND_RUNTIME_MODE = "authored_usd"`
- `HAND_USD_OVERRIDES["right"] = "/abs/path/to/new_right_dex3_1.usd"`

and similarly for the left side when validating left-hand batches.

---

## Minimal Checklist For A New Hand

When adding a new hand family, this is the minimum checklist:

1. Put left/right USD and URDF assets under `assets/hands/<hand_name>/asset/`
2. Write `config/left_hand.yaml`
3. Write `config/right_hand.yaml`
4. Write `collision/left_collision.yaml`
5. Write `collision/right_collision.yaml`
6. Write `config/contact_points.yaml`
7. Write `config/pose_seeds.yaml`
8. Confirm that semantic points reference valid collision spheres
9. Run single-hand anchor preview
10. Run single-hand seed preview
11. Run single-hand staged preview
12. Run bimanual preview

For `cat1`, `cat2`, `cat3`, and `cat4`, the runtime now scales over all configured non-thumb fingers, so finger count itself should not require special-case code as long as the semantic naming stays compatible.

---

## Notes

- `run_single_hand.py` is for anchor proposal / anchor visualization.
- `run_single_hand_seed_preview.py` is for initial seeds only.
- `run_single_hand_staged_pose_preview.py` is for optimized poses.
- `run_bimanual.py` is the main bimanual entrypoint.
- `run_bimanual.py` supports `--max_workers` for parallel per-anchor side-bundle solves.
- The first run can still be expensive, especially for `cat2`, because `cat2` has a heavier optimization problem than `cat1`.
