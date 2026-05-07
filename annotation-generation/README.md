# Code Organized Workspace

This workspace contains two related pieces:

- `annotation_generation/`: standalone Isaac Sim pipelines for generating, previewing, and validating grasp and articulation annotations.
- `MagicSim/`: a simulation application package with robot, task, scene, and asset configuration.

The shared goal is to produce reusable robot manipulation annotations and simulation assets without committing user-specific paths, credentials, or local runtime metadata.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `annotation_generation/Grasp/` | Functional grasp generation for object assets. |
| `annotation_generation/DexGrasp/` | Dexterous grasp physics validation for hand-specific grasp annotations. |
| `annotation_generation/Bigripper/` | Bi-gripper grasp generation, anchor search, and re-evaluation. |
| `annotation_generation/Bimanual_dexhand/` | Bimanual and single-hand dexterous grasp proposal, staged optimization, SDF generation, previews, and validation exports. |
| `annotation_generation/Door/` | Door push and pull grasp validation. |
| `annotation_generation/Articulation/` | Drawer, edge-opening, handle-opening, close-by-push, and rotate/push trajectory generation. |
| `demo_assets/` | Small, shareable assets for local smoke tests and examples. |
| `MagicSim/` | Main MagicSim app, scene/task configs, robot configs, assets, Dockerfiles, and robot import scripts. |
| `tmp/` | Scratch or migration material. Do not treat as canonical source unless explicitly promoted. |

## Demo Assets

These assets are intended for examples and quick checks:

| Category | Assets |
| --- | --- |
| Objects | `demo_assets/Object/Bin`, `demo_assets/Object/bottle_2`, `demo_assets/Object/apple_001`, `demo_assets/Object/ball_013` |
| Drawers | `demo_assets/Drawer/7120`, `demo_assets/Drawer/46230` |
| Door push | `demo_assets/door_push` |
| Garment | `demo_assets/Garment` |

Annotations generally live beside their asset in an `Annotation/` directory, or as category-specific JSON files next to `Object.usd`.

## Privacy And Path Policy

Do not commit machine-specific paths, private remotes, credentials, local usernames, personal emails, access tokens, simulator logs, bytecode caches, or generated runtime caches.

Use one of these instead:

- Relative paths from this workspace.
- Environment variables for user-provided paths.
- Empty or anonymous metadata fields when a value is not required.
- Generated output directories that can be deleted and rebuilt.

Generated directories such as `.cache/`, `__pycache__/`, simulator logs, and temporary validation output should stay out of any shared artifact unless intentionally scrubbed.

## Environment Setup

Most annotation scripts import Isaac Sim and should be launched with the Python interpreter that matches your Isaac Sim installation.

Set these paths per shell session:

```bash
cd <workspace>
export WORKSPACE_ROOT="$(pwd)"
export ANNOTATION_ROOT="$WORKSPACE_ROOT/annotation_generation"
export DEMO_ASSETS_ROOT="$WORKSPACE_ROOT/demo_assets"
export MAGICSIM_HOME="$WORKSPACE_ROOT/MagicSim"
export MAGICSIM_ASSETS="$MAGICSIM_HOME/Assets"
export ISAAC_SIM_PYTHON="<path-to-isaac-sim-python>"
```

For MagicSim development, use the package tools in `MagicSim/` and keep dependency caches outside committed source. Dockerfiles accept configurable user and login build arguments, so fixed credentials do not need to be stored in source.

## Common Environment Variables

The scripts are designed to accept user paths through environment variables or CLI arguments.

| Variable | Used for |
| --- | --- |
| `GRASP_INPUT_ROOT` | Object root for `Grasp/Object_grasp_pose_pipeline.py`. |
| `GRASP_GRIPPER_USD` | Gripper USD for functional grasp generation. |
| `GRASP_DATASET_PATH` | Dataset root for functional grasp generation. |
| `GRASP_FUNCTIONAL_PAIRS_PATH` | Functional part-pair JSON. |
| `BIGRIPPER_GRIPPER_USD` | Gripper USD for bi-gripper generation. |
| `SHARPA_ROBOT_USD`, `SHARPA_OBJECT_USD`, `SHARPA_GRASP_JSON` | Sharpa validation inputs. |
| `BIMANUAL_LEFT_HAND_USD`, `BIMANUAL_RIGHT_HAND_USD`, `BIMANUAL_OBJECT_USD` | Bimanual validation inputs. |
| `SINGLE_HAND_LEFT_USD`, `SINGLE_HAND_RIGHT_USD`, `SINGLE_HAND_OBJECT_USD` | Single-hand validation inputs. |
| `DOOR_PUSH_OBJECT_USD`, `DOOR_PULL_OBJECT_USD` | Door validation objects. |
| `OPEN_HANDLE_*`, `OPEN_EDGE_*`, `CLOSE_PUSH_*`, `ROTATE_*` | Articulation trajectory assets and datasets. |
| `MAGICSIM_DWB_DEBUG_PATH` | Debug plot output path for the MagicSim DWB planner. |

## Bimanual Dexhand Workflow

Build or refresh an object SDF cache:

```bash
"$ISAAC_SIM_PYTHON" annotation_generation/Bimanual_dexhand/run_build_object_sdf.py \
  --object_usd demo_assets/Object/Bin/Object.usd \
  --cache_path outputs/sdf/bin_object_sdf.npz \
  --headless
```

Run a single-hand proposal:

```bash
"$ISAAC_SIM_PYTHON" annotation_generation/Bimanual_dexhand/run_single_hand.py \
  --hand_dir annotation_generation/Bimanual_dexhand/assets/hands/sharpa \
  --config_dir annotation_generation/Bimanual_dexhand/configs \
  --object_usd demo_assets/Object/Bin/Object.usd \
  --side right \
  --category cat1 \
  --headless \
  --save_json outputs/single_hand_bin_right.json
```

Run a bimanual staged export:

```bash
"$ISAAC_SIM_PYTHON" annotation_generation/Bimanual_dexhand/run_bimanual.py \
  --hand_dir annotation_generation/Bimanual_dexhand/assets/hands/sharpa \
  --config_dir annotation_generation/Bimanual_dexhand/configs \
  --object_usd demo_assets/Object/Bin/Object.usd \
  --primary_side right \
  --category cat1 \
  --headless \
  --save_format both \
  --save_json outputs/bimanual_bin.json
```

## Other Annotation Pipelines

Functional object grasp generation:

```bash
GRASP_INPUT_ROOT=demo_assets/Object \
GRASP_DATASET_PATH=demo_assets/Object \
GRASP_FUNCTIONAL_PAIRS_PATH=annotation_generation/Grasp/functional_list.json \
"$ISAAC_SIM_PYTHON" annotation_generation/Grasp/Object_grasp_pose_pipeline.py
```

Bi-gripper generation:

```bash
"$ISAAC_SIM_PYTHON" annotation_generation/Bigripper/bi_gripper_pipeline.py \
  --object_usd demo_assets/Object/Bin/Object.usd \
  --object_type Bin
```

Dexterous validation scripts are hand-specific:

```bash
"$ISAAC_SIM_PYTHON" annotation_generation/DexGrasp/sharpa_physics_validation_parallel.py
"$ISAAC_SIM_PYTHON" annotation_generation/DexGrasp/dex3_1_physics_validation.py
"$ISAAC_SIM_PYTHON" annotation_generation/DexGrasp/xhand_physics_validation_parallel.py
```

Door and articulation scripts follow the same pattern: provide assets with environment variables or relative defaults, run with Isaac Sim Python, and save annotations beside the asset or under an explicit output directory.

## MagicSim Notes

`MagicSim/` contains the application code, assets, task configs, robot configs, Dockerfiles, and robot import utilities.

Useful entry points:

```bash
cd MagicSim
python Script/Robot/import_new_robot.py --help
python -m Script.test_sharpa_curobo_ik
```

Scene and task configs live under `MagicSim/src/magicsim/Task/` and `MagicSim/src/magicsim/Env/Conf/`. Prefer `$MAGICSIM_ASSETS` or paths relative to `MagicSim/` when adding new assets.

## Output Conventions

Use these conventions for generated artifacts:

- Put throwaway outputs under `outputs/` or a task-specific output directory.
- Keep object-local annotations near the asset under `Annotation/`.
- Keep SDF caches under an explicit cache or output directory.
- Do not commit simulator runtime caches or bytecode caches.
- Use anonymous author and repository metadata before sharing.

## Pre-Share Checklist

Before publishing or sending this workspace:

1. Remove generated caches and bytecode directories.
2. Scan text and binary files for local paths, usernames, emails, tokens, and private remotes.
3. Inspect archives and compressed numpy files, because they can embed metadata.
4. Check nested Git repositories for author metadata and remotes.
5. Re-run JSON validation after editing annotation files.
6. Run a small smoke test on at least one demo asset per workflow you changed.

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| Isaac Sim imports fail | Launch the script with the Isaac Sim Python interpreter for your installation. |
| Asset cannot be found | Replace absolute constants with relative paths, CLI arguments, or environment variables. |
| Empty or stale SDF results | Rebuild with `--force_rebuild` and write to a clean cache path. |
| Validation output is slow | Use `--headless`, reduce surface samples, or use the safe export profile where supported. |
| Share scan finds generated logs | Delete caches/logs and regenerate them locally when needed. |

