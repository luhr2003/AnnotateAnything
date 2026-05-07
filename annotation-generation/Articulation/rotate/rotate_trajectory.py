import asyncio
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Set

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils import transformations as transform_utils
from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdPhysics, Usd, PhysxSchema, UsdLux, PhysicsSchemaTools
import omni.usd
import omni.kit.app
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.cloner import GridCloner 
from omni.timeline import get_timeline_interface
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
    ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
timeline = get_timeline_interface()
import isaacsim.replicator.grasping.transform_utils as transform_utils
from omni.physx import get_physx_scene_query_interface, get_physx_interface

# Constants
APPROACH_DISTANCE = 0.08  # Distance to step back before approaching (m)
MOVE_STEPS = 100          # Steps for linear interpolation
CLOSE_STEPS = 30         # Steps for closing gripper
HOLD_STEPS = 20          # Extra physics steps after closing to let contacts settle
TRAJECTORY_SIM_STEPS_PER_WAYPOINT = 3  # Physics steps per trajectory waypoint
NUM_COPIES = 50         # Number of copies to create in cloner
CLONE_SPACING = 5.0    # Spacing between clones in cloner grid (m)
TRAJ_OVERSHOOT_CHECKS   = 5 #TBD
GRIPPER_FINGERTIP_OFFSET = 0.195

# Gripper approach axis used for filtering (local axis in gripper pose frame)
GRIPPER_APPROACH_LOCAL = (0, 0, 1)

APPROACH_POSITION_THRESHOLD = 0.005
JOINT_SUCCESS_THRESHOLD = 0.95

BUTTON_ROTATE_ANGLE = np.radians(45)

#Target Points Sampling Constants
GRIPPER_MAX_APERTURE = 0.08   
BUTTON_CAP_THICKNESS = 0.002
NUM_BUTTON_GRASP_ROLLS = 8 
NUM_POINTS_SAMPLED_PER_BUTTON = 200
FRACTION_TO_KEEP_AROUND_BUTTON_CENTER = 0.05

# =======================
# Processing Mode Configuration
# =======================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parents[1]

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

PROCESSING_MODE = "single"  # Options: "single" or "dataset"

# For single object mode
SINGLE_OBJECT_USD = _path_from_env("ROTATE_OBJECT_USD", _THIS_DIR / "102055" / "Object.usd")

INPUT_DATASET_PATH = _path_from_env("ROTATE_DATASET_PATH", _THIS_DIR)
GRIPPER_USD = _path_from_env("ROTATE_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")
LOG_FILE = INPUT_DATASET_PATH / "close_by_push_completed_objects.txt"

#Isaac Sim Stage Paths
ENV_BASE_PATH = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"
PHYSICS_SCENE_PATH = "/World/physicsScene"

def env_path(i: int) -> str:
    # env paths are /World/Envs/env_0, /World/Envs/env_1, ...
    return f"{ENV_ROOT_PREFIX}_{i}"

def obj_wrap(i: int) -> str:
    return f"{env_path(i)}/Object"

def obj_ref(i: int) -> str:
    return f"{env_path(i)}/Object/ref"

def grip_wrap(i: int) -> str:
    return f"{env_path(i)}/Flying_hand_probe_pro"

def grip_ref(i: int) -> str:
    return f"{env_path(i)}/Flying_hand_probe_pro/ref"

def grip_base(i: int) -> str:
    return f"{grip_ref(i)}/panda_hand"

OBJECT_WRAPPER_PATH = obj_wrap(0)
OBJECT_REF_PATH = obj_ref(0)
GRIPPER_WRAPPER_PATH = grip_wrap(0)
GRIPPER_REF_PATH = grip_ref(0)

# =======================
# Helper Functions
# =======================

def get_completed_objects(log_file: Path) -> set:
    """Read list of already processed objects from log file."""
    if not log_file.exists():
        return set()
    
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def mark_object_completed(log_file: Path, obj_id: str):
    """Append object ID to completion log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"{obj_id}\n")


def find_all_objects(dataset_path: Path) -> List[Tuple[Path, str]]:
    """
    Find all Object.usd files in the dataset.
    
    Returns:
        List of (obj_usd_path, obj_id) tuples
    """
    objects = []
    
    # Iterate through all subdirectories
    for obj_dir in sorted(dataset_path.iterdir()):
        if not obj_dir.is_dir():
            continue
        
        obj_usd = obj_dir / "Object.usd"
        
        if obj_usd.exists():
            obj_id = obj_dir.name
            objects.append((obj_usd, obj_id))
    
    return objects

async def step_simulation(steps: int):
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()

async def ensure_timeline_playing():
    """Ensure timeline is playing - force restart if stopped"""
    if not timeline.is_playing():
        print(f"[DEBUG] Timeline stopped, restarting...")
        timeline.play()
        await omni.kit.app.get_app().next_update_async()

def setup_physics_scene(stage):
    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")

    if not prim.HasAPI(UsdPhysics.Scene):
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    else:
        scene = UsdPhysics.Scene(prim)
    
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
        
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx_scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(32768)
    physx_scene_api.CreateGpuTotalAggregatePairsCapacityAttr().Set(32768)
    return PHYSICS_SCENE_PATH

def get_bbox_bottom_center(stage, prim_path: str):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()
    
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    
    bottom_center = Gf.Vec3d(
        (min_point[0] + max_point[0]) / 2.0,
        (min_point[1] + max_point[1]) / 2.0,
        min_point[2]
    )
    
    return bottom_center


def disable_instanceable_for_grasp_generation(stage, object_ref_path: str) -> List[str]:
    """Disable instanceable on ALL prims under object_ref_path.

    Many PartNet assets mark nested prims (links/visuals/World/mesh) as instanceable,
    not necessarily the reference prim itself. Disabling only the reference prim is often a no-op.

    Returns:
        changed_paths: list of prim path strings that were changed from instanceable=True to False.
        Use this list to restore instanceable later.
    """
    root = stage.GetPrimAtPath(object_ref_path)

    if not root.IsValid():
        print(f"[ERROR] Invalid object reference path: {object_ref_path}")
        return []

    changed_paths: List[str] = []

    # Disable instanceable across the entire subtree.
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue
        if prim.IsInstanceable():
            try:
                prim.SetInstanceable(False)
                changed_paths.append(prim.GetPath().pathString)
            except Exception as e:
                print(f"[WARN] Failed to SetInstanceable(False) on {prim.GetPath()}: {e}")

    print(f"[INFO] Disabled instanceable on {len(changed_paths)} prim(s) under {object_ref_path}")
    return changed_paths


def restore_instanceable(stage, changed_paths: List[str]):
    """Restore original instanceable state on prims we temporarily made editable."""
    if not changed_paths:
        print("[INFO] No instanceable prims to restore")
        return

    # Restore deepest prims first so parent instanceability does not interfere
    # with restoring authored state on descendants.
    unique_paths = sorted(set(changed_paths), key=lambda p: (p.count("/"), p), reverse=True)

    cleared = 0
    restored = 0
    fallback_restored = 0
    unresolved: List[str] = []

    for p in unique_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue
        try:
            if prim.HasAuthoredMetadata("instanceable"):
                prim.ClearInstanceable()
                cleared += 1
        except Exception as e:
            print(f"[WARN] Failed to clear instanceable override on {p}: {e}")
            unresolved.append(p)

    for p in unique_paths:
        if p in unresolved:
            continue
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue
        if prim.IsInstanceable():
            restored += 1
            continue
        try:
            prim.SetInstanceable(True)
            if prim.IsInstanceable():
                fallback_restored += 1
            else:
                unresolved.append(p)
        except Exception as e:
            print(f"[WARN] Failed to restore instanceable on {p}: {e}")
            unresolved.append(p)

    total_restored = restored + fallback_restored
    print(
        f"[INFO] Restored instanceable on {total_restored}/{len(unique_paths)} prim(s) "
        f"(cleared {cleared} local override(s), fallback-authored {fallback_restored})"
    )
    if unresolved:
        sample = ", ".join(unresolved[:5])
        if len(unresolved) > 5:
            sample += ", ..."
        print(f"[WARN] Instanceable still not restored on {len(unresolved)} prim(s): {sample}")

def save_trajectories_to_json(
    trajectories: List[Dict],
    obj_usd_path: Path,
    bottom_center: List[float]
) -> Path:
    """
    Save validated trajectories to JSON format grouped by joint.
    
    Args:
        trajectories: List of validated trajectory dicts from physics_validation_loop
        obj_usd_path: Path to Object.usd file
        bottom_center: [x, y, z] bottom center coordinates
    
    Returns:
        Path to saved JSON file
    """
    # Extract obj_id from path
    obj_id = obj_usd_path.parent.name
    
    # For dataset mode with category structure
    if PROCESSING_MODE == "dataset":
        try:
            obj_cat = obj_usd_path.parent.parent.name
            type_str = f"{obj_cat}/{obj_id}/Object.usd"
        except:
            type_str = f"{obj_id}/Object.usd"
    else:
        # Single object mode
        obj_cat = obj_usd_path.parent.parent.name
        type_str = f"{obj_id}/Object.usd"
    
    # Create Annotation directory at same level as Object.usd
    annotation_dir = obj_usd_path.parent / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect initial joint angles keyed by joint_name
    # Revolute: convert radians -> degrees. Prismatic: keep meters as-is.
    initial_joint_angles: Dict[str, float] = {}
    for traj in trajectories:
        jname = traj.get("joint_name", "unknown")
        jtype = traj.get("joint_type", "revolute")
        if jname not in initial_joint_angles:
            init_pos = traj.get("initial_joint_pos", None)
            if init_pos is not None:
                if jtype == "revolute":
                    initial_joint_angles[jname] = float(np.degrees(init_pos))
                else:
                    initial_joint_angles[jname] = float(init_pos)

    # Build JSON structure
    data = {
        "type": obj_cat,
        "bottom_center": {
            "x": float(bottom_center[0]),
            "y": float(bottom_center[1]),
            "z": float(bottom_center[2])
        },
        "initial_joint_angles": initial_joint_angles,
        "trajectories": {}
    }
    
    # Group trajectories by joint
    trajectories_by_joint = {}
    
    # Process each trajectory
    for idx, traj in enumerate(trajectories):
        trajectory_positions = np.asarray(traj["trajectory_positions"], dtype=np.float64)
        trajectory_orientations = np.asarray(traj["trajectory_orientations"], dtype=np.float64)
        
        # Get joint information
        joint_name = traj.get("joint_name", "unknown")
        
        # Check if trajectory terminated early
        termination_step = traj.get("termination_step", None)
        if termination_step is not None:
            # Truncate to termination step (inclusive, so +1)
            trajectory_positions = trajectory_positions[:termination_step + 1]
            trajectory_orientations = trajectory_orientations[:termination_step + 1]
            print(f"[INFO] Trajectory {idx} ({joint_name}): "
                  f"Truncated to {termination_step + 1} waypoints (early termination at step {termination_step})")
        
        # Build waypoint list: each waypoint is [x, y, z, qw, qx, qy, qz]
        waypoints = []
        for pos, quat in zip(trajectory_positions, trajectory_orientations):
            # quat is [w, x, y, z] from your code
            waypoint = [
                float(pos[0]),  # x
                float(pos[1]),  # y
                float(pos[2]),  # z
                float(quat[0]), # w
                float(quat[1]), # x
                float(quat[2]), # y
                float(quat[3])  # z
            ]
            waypoints.append(waypoint)
        
        # Group by joint_name
        if joint_name not in trajectories_by_joint:
            trajectories_by_joint[joint_name] = []
        trajectories_by_joint[joint_name].append(waypoints)
    
    # Build the nested structure: joint_{num} -> { "1": [...], "2": [...], ... }
    for joint_name, joint_trajectories in trajectories_by_joint.items():
        data["trajectories"][joint_name] = {}
        for traj_num, waypoints in enumerate(joint_trajectories, start=1):
            data["trajectories"][joint_name][str(traj_num)] = waypoints
    
    # Save to JSON
    json_path = annotation_dir / "rotate_trajectory.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[INFO] Saved {len(trajectories)} trajectories to {json_path}")
    print(f"[INFO] Trajectories organized by joint:")
    for joint_name, joint_trajectories in trajectories_by_joint.items():
        print(f"  - {joint_name}: {len(joint_trajectories)} trajectories")
    
    return json_path

# =======================
# Fix Object Base to World (FixedJoint)
# =======================
def fix_object_base_to_world(stage, object_ref_path: str, base_link_name: str = "base",
                             joint_path: str = "/World/ObjectFixedToWorld"):
    base_path = f"{object_ref_path}/{base_link_name}"
    base_prim = stage.GetPrimAtPath(base_path)

    if not base_prim.IsValid():
        root = stage.GetPrimAtPath(object_ref_path)
        if not root.IsValid():
            print(f"[WARN] fix_object_base_to_world: invalid object ref path {object_ref_path}")
            return
        candidate = None
        for prim in root.GetChildren():
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                candidate = prim
                break
        if candidate is None:
            print(f"[WARN] fix_object_base_to_world: could not find base rigid body under {object_ref_path}")
            return
        base_prim = candidate
        base_path = candidate.GetPath().pathString

    if not base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"[WARN] fix_object_base_to_world: base prim is not a rigid body: {base_path}")
        return

    # Base world pose
    xformable = UsdGeom.Xformable(base_prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_xf.ExtractTranslation()
    qd = world_xf.ExtractRotation().GetQuat()  # usually Quatd

    # Remove existing joint prim if present
    if stage.GetPrimAtPath(joint_path).IsValid():
        try:
            stage.RemovePrim(Sdf.Path(joint_path))
        except Exception:
            pass

    fj = UsdPhysics.FixedJoint.Define(stage, joint_path)

    # Body0 = world (leave empty), Body1 = base
    fj.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])

    # Joint frame on WORLD: at current base world pose
    fj.CreateLocalPos0Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
    fj.CreateLocalRot0Attr().Set(Gf.Quatf(
        float(qd.GetReal()),
        Gf.Vec3f(float(qd.GetImaginary()[0]), float(qd.GetImaginary()[1]), float(qd.GetImaginary()[2]))
    ))

    # Joint frame on BASE: base origin (identity)
    fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    print(f"[INFO] Fixed object base to world at current pose: base={base_path} joint={joint_path}")

# =======================
# Grasp Generation
# =======================
def plane_normal_pca(points_world: np.ndarray) -> Optional[np.ndarray]:
    """Fit a plane by PCA; return unit normal (smallest eigenvector)."""
    if points_world is None or points_world.shape[0] < 200:
        return None

    centroid = points_world.mean(axis=0)
    X = points_world - centroid
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)

    # eigenvalues ascending; smallest eigenvector is plane normal
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n

def sample_mesh_surface_points(stage, mesh_path, n_points=500):
    """Sample points uniformly on mesh surface using area-weighted triangle sampling.
    
    Args:
        stage: USD stage
        mesh_path: Path to mesh prim
        n_points: Number of points to sample
    
    Returns:
        [n_points, 3] array of world-space surface points
    """
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        print(f"[WARNING] Invalid mesh prim at {mesh_path}")
        return np.zeros((0, 3), dtype=np.float64)
    
    mesh = UsdGeom.Mesh(prim)

    verts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    
    if not verts or not counts or not indices:
        print(f"[WARNING] Incomplete mesh data at {mesh_path}")
        return np.zeros((0, 3), dtype=np.float64)
    
    verts = np.asarray(verts, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)

    # Check if mesh is triangulated (all faces are triangles)
    if not np.all(counts == 3):
        print(f"[WARNING] Mesh at {mesh_path} is not triangulated (has non-triangle faces)")
        # Filter to only use triangular faces
        tri_mask = counts == 3
        if not np.any(tri_mask):
            print(f"[ERROR] No triangular faces found in mesh")
            return np.zeros((0, 3), dtype=np.float64)
        
        # Rebuild indices for triangles only
        new_indices = []
        idx_ptr = 0
        for i, count in enumerate(counts):
            if count == 3:
                new_indices.extend(indices[idx_ptr:idx_ptr+3])
            idx_ptr += count
        indices = np.array(new_indices, dtype=np.int32)
    
    # Reshape to triangles
    try:
        tris = verts[indices].reshape(-1, 3, 3)
    except Exception as e:
        print(f"[ERROR] Failed to reshape mesh vertices: {e}")
        return np.zeros((0, 3), dtype=np.float64)

    if len(tris) == 0:
        print(f"[WARNING] No triangles in mesh at {mesh_path}")
        return np.zeros((0, 3), dtype=np.float64)

    # Area-weighted sampling
    edge1 = tris[:,1] - tris[:,0]
    edge2 = tris[:,2] - tris[:,0]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    
    total_area = areas.sum()
    if total_area < 1e-12:
        print(f"[WARNING] Mesh at {mesh_path} has near-zero total area")
        return np.zeros((0, 3), dtype=np.float64)
    
    probs = areas / total_area

    # Choose triangles
    tri_idx = np.random.choice(len(tris), min(n_points, len(tris)), p=probs)
    t = tris[tri_idx]

    # Barycentric random points
    n_actual = len(tri_idx)
    u = np.random.rand(n_actual, 1)
    v = np.random.rand(n_actual, 1)
    mask = u + v > 1
    u[mask], v[mask] = 1-u[mask], 1-v[mask]
    w = 1 - u - v

    pts = t[:,0] * w + t[:,1] * u + t[:,2] * v

    # Transform to world
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts_world = np.asarray([_gvec_to_np(xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in pts], dtype=np.float64)

    return pts_world

def find_continuous_button_joints_in_urdf(urdf_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """Parse URDF to find continuous joints treated as buttons.

    Returns:
        List of (joint_name, child_link_name, joint_axis)
    """
    button_joints: List[Tuple[str, str, np.ndarray]] = []

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for joint in root.findall("joint"):
            joint_type = joint.get("type", "").lower()
            if joint_type != "continuous":
                continue

            joint_name = joint.get("name", "")
            child_link = joint.find("child")
            child_link_name = child_link.get("link") if child_link is not None else None
            if not child_link_name:
                continue

            axis_elem = joint.find("axis")
            if axis_elem is not None:
                xyz_str = axis_elem.get("xyz", "0 0 1")
                axis_vals = [float(x) for x in xyz_str.split()]
                joint_axis = np.array(axis_vals, dtype=np.float64)
            else:
                joint_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            if np.linalg.norm(joint_axis) < 1e-9:
                joint_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            button_joints.append((joint_name, child_link_name, joint_axis))
            print(f"[DEBUG] Found button joint: {joint_name} (continuous) -> {child_link_name}, axis={joint_axis}")

    except Exception as e:
        print(f"[ERROR] Failed to parse URDF for button joints: {e}")

    return button_joints

def collect_all_child_link_surface_samples(stage, object_ref_path: str, child_link_name: str, n_points_per_mesh: int = 500) -> np.ndarray:
    """Sample surface points from ALL visual geometries under a child link using area-weighted sampling.
    
    Args:
        stage: USD stage
        object_ref_path: Path to object reference (e.g., /World/Envs/env_0/Object/ref)
        child_link_name: Name of the child link (e.g., 'link_0')
        n_points_per_mesh: Number of points to sample per mesh
    
    Returns:
        [N, 3] array of world-space surface sample points from all meshes under the child link
    """
    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    
    # Path to the child link's visual geometry
    # Typical structure: {asset_root}/{child_link_name}/visuals
    visuals_root_path = f"{asset_root}/{child_link_name}/visuals"
    
    visuals_prim = stage.GetPrimAtPath(visuals_root_path)
    if not visuals_prim.IsValid():
        print(f"[WARNING] No visuals found at {visuals_root_path}")
        return np.zeros((0, 3), dtype=np.float64)
    
    # Ensure payloads are loaded
    try:
        visuals_prim.Load()
    except Exception:
        pass
    
    all_sampled_points = []
    mesh_count = 0
    
    # Collect and sample points from all mesh prims under visuals
    for prim in Usd.PrimRange(visuals_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh_count += 1
        mesh_path = prim.GetPath().pathString
        
        # Use area-weighted surface sampling
        sampled_pts = sample_mesh_surface_points(stage, mesh_path, n_points=n_points_per_mesh)
        
        if sampled_pts.shape[0] > 0:
            all_sampled_points.append(sampled_pts)
            print(f"[INFO]   Sampled {sampled_pts.shape[0]} points from mesh: {prim.GetName()}")
    
    if not all_sampled_points:
        print(f"[WARNING] No mesh points sampled from {visuals_root_path} ({mesh_count} meshes found)")
        return np.zeros((0, 3), dtype=np.float64)
    
    all_sampled_points = np.concatenate(all_sampled_points, axis=0)
    print(f"[INFO] Total sampled points from {child_link_name}: {all_sampled_points.shape[0]} (from {mesh_count} meshes)")
    
    return all_sampled_points

def compute_object_center(stage, object_wrapper_path: str) -> np.ndarray:
    """Compute center of entire object."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(object_wrapper_path)
    
    if not prim.IsValid():
        return np.array([0, 0, 0])
    
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()
    center = bbox_range.GetMidpoint()
    
    return np.array([float(center[0]), float(center[1]), float(center[2])])

def get_link_local_bbox_center(stage, link_path: str) -> Optional[np.ndarray]:
    """Get center of link's local bounding box in world coordinates."""
    link_prim = stage.GetPrimAtPath(link_path)
    
    if not link_prim.IsValid():
        return None
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    local_bbox = bbox_cache.ComputeLocalBound(link_prim)
    local_range = local_bbox.GetRange()
    local_center = local_range.GetMidpoint()
    
    xformable = UsdGeom.Xformable(link_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_center = world_transform.Transform(local_center)
    
    return np.array([float(world_center[0]), float(world_center[1]), float(world_center[2])])


# =======================
# Door surface normal from geometry (door mesh plane) + handle sign
# =======================

def _gvec_to_np(v) -> np.ndarray:
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)

def _world_xf_of_prim(prim) -> Gf.Matrix4d:
    xf = UsdGeom.Xformable(prim)
    return xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

def resolve_asset_root_under_ref(stage, object_ref_path: str) -> str:
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    if not ref_prim.IsValid():
        return object_ref_path

    # Go one level deeper unconditionally: pick the first valid child that
    # itself has children (the scene root), regardless of its name.
    # This handles partnet_* and any other naming convention.
    children = [c for c in ref_prim.GetChildren() if c.IsValid()]
    if len(children) == 1:
        return children[0].GetPath().pathString
    for child in children:
        if list(child.GetChildren()):
            return child.GetPath().pathString
    # Fallback: ref IS the root (no wrapper layer present)
    return object_ref_path

def collect_mesh_points_world(stage, root_path: str, max_points: int = 6000) -> np.ndarray:
    """Collect (subsampled) mesh vertices under root_path, transformed to world.

    Works for paths like:
      .../visuals/visual_mesh_0/World/mesh
    and higher-level roots like:
      .../link_0/visuals

    Also attempts to load payloads so PrimRange can see Mesh prims.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return np.zeros((0, 3), dtype=np.float64)

    # Ensure payloads are loaded (safe no-op if none)
    try:
        root_prim.Load()
    except Exception:
        pass

    pts_all = []

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue

        xf = _world_xf_of_prim(prim)
        pts_np = np.asarray([_gvec_to_np(xf.Transform(p)) for p in pts], dtype=np.float64)
        pts_all.append(pts_np)

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float64)

    pts_all = np.concatenate(pts_all, axis=0)

    if pts_all.shape[0] > max_points:
        idx = np.random.choice(pts_all.shape[0], size=max_points, replace=False)
        pts_all = pts_all[idx]

    return pts_all

# =======================
# Button helpers (continuous joints)
# =======================
def compute_button_normal_from_geometry(
    surface_points: np.ndarray,
    object_center: np.ndarray,
    button_center: np.ndarray,
    fallback_axis: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Derive button press normal purely from surface geometry via PCA plane fit.
    
    The button surface points should lie approximately on a flat cap — the plane
    normal (smallest PCA eigenvector) is the press axis.
    
    Sign is chosen so the normal points from object center toward button center
    (i.e., outward, the press direction).
    
    Falls back to fallback_axis (e.g., joint axis) if too few points for PCA.
    """
    n = None

    if surface_points is not None and surface_points.shape[0] >= 10:
        n = plane_normal_pca(surface_points)  # already defined, needs >=200 pts

    # plane_normal_pca requires 200+ points — relax for cap points which may be fewer
    if n is None and surface_points is not None and surface_points.shape[0] >= 10:
        # Inline PCA without the 200-point guard
        centroid = surface_points.mean(axis=0)
        X = surface_points - centroid
        cov = (X.T @ X) / max(X.shape[0] - 1, 1)
        w, v = np.linalg.eigh(cov)
        n = v[:, 0]  # smallest eigenvalue = plane normal
        n = n / (np.linalg.norm(n) + 1e-12)

    if n is None:
        if fallback_axis is not None:
            print("[WARN] compute_button_normal_from_geometry: too few points, falling back to joint axis")
            n = np.asarray(fallback_axis, dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)
        else:
            print("[ERROR] compute_button_normal_from_geometry: no points and no fallback")
            return None

    # Disambiguate sign: normal should point outward (object center → button center)
    obj_to_button = np.asarray(button_center, dtype=np.float64) - np.asarray(object_center, dtype=np.float64)
    if np.dot(n, obj_to_button) < 0.0:
        n = -n

    return n / (np.linalg.norm(n) + 1e-12)

def get_button_surface_center(
    surface_points: np.ndarray,
    button_normal: np.ndarray,
    epsilon: float = 0.003,  # 3mm slab around the peak
) -> Optional[np.ndarray]:
    """Return button surface center: centroid of points within epsilon of the
    furthest projection along the normal. More robust than single argmax point."""
    if surface_points is None or surface_points.shape[0] == 0:
        return None

    n = np.asarray(button_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    proj = surface_points @ n
    max_proj = float(proj.max())

    # All points within epsilon of the peak projection
    cap_mask = proj >= (max_proj - epsilon)
    cap_points = surface_points[cap_mask]

    if cap_points.shape[0] == 0:
        return None

    # Centroid of the cap slab
    return cap_points.mean(axis=0)

def generate_button_grasp_quats(
    button_normal: np.ndarray,
    num_poses: int = NUM_BUTTON_GRASP_ROLLS,
) -> np.ndarray:
    """Generate grasp quaternions by rolling around the button normal.

    - Gripper local GRIPPER_APPROACH_LOCAL aligns with -button_normal
    - num_poses different roll angles about that local axis
    - Position is NOT changed here, only orientation.
    """
    n = np.asarray(button_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Local approach axis in gripper frame (e.g., (0,0,1))
    approach_local = np.asarray(GRIPPER_APPROACH_LOCAL, dtype=np.float64)
    approach_local = approach_local / (np.linalg.norm(approach_local) + 1e-12)

    # In world frame, approach axis should point opposite the surface normal
    # i.e., rot(approach_local) = -n
    z_world = -n  # "approach" axis in world

    # Build an orthonormal basis where approach_local maps to z_world
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, z_world)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_world = np.cross(tmp, z_world)
    x_world /= (np.linalg.norm(x_world) + 1e-12)
    y_world = np.cross(z_world, x_world)
    y_world /= (np.linalg.norm(y_world) + 1e-12)

    # Rotation matrix whose columns are gripper local axes in world:
    # assuming gripper local basis (X_local, Y_local, Z_local) where
    # Z_local = GRIPPER_APPROACH_LOCAL.
    # For now we treat the local frame as canonical and just ensure Z matches.
    R_base = np.column_stack([x_world, y_world, z_world])

    quats: List[np.ndarray] = []
    rng = np.random.default_rng()

    for _ in range(num_poses):
        roll = float(rng.uniform(0.0, 2.0 * np.pi))
        # Roll around the gripper's local approach axis, which in world is z_world
        R_roll = R.from_rotvec(roll * np.array([0.0, 0.0, 1.0], dtype=np.float64)).as_matrix()
        R_world = R_base @ R_roll
        r_obj = R.from_matrix(R_world)
        q_xyzw = r_obj.as_quat()  # [x,y,z,w]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        quats.append(q_wxyz)

    return np.asarray(quats, dtype=np.float64)

# =======================
# Quaternion / approach helpers for filtering
# =======================
def generate_button_grasps_for_all_joints(
    stage,
    dataset_dir: str,
    object_wrapper_path: str,
    object_ref_path: str,
    gripper_wrapper_path: str,
    num_rolls_per_button: int = NUM_BUTTON_GRASP_ROLLS,
) -> Dict[str, Dict]:
    """Generate grasp poses for all continuous (button) joints.

    For each button joint:
      - compute ONE button surface center point
      - compute ONE approach normal
      - generate num_rolls_per_button quaternions by rolling around local Z
      - position is the same for all quaternions

    Returns:
        Dict[joint_name] = {
            'grasp_poses': [(pos[3], quat[4]), ...],
            'joint_type': 'continuous',
            'child_link_name': str,
            'joint_axis': np.ndarray,
            'button_normal': np.ndarray,
            'button_center': np.ndarray,
            'radius': float,
        }
    """
    urdf_path = os.path.join(dataset_dir, "mobility.urdf")

    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF not found at {urdf_path}")
        return {}

    button_joints = find_continuous_button_joints_in_urdf(urdf_path)
    if not button_joints:
        print("[INFO] No continuous button joints found in URDF")
        return {}

    print(f"\n[INFO] Found {len(button_joints)} continuous (button) joint(s)")

    all_button_grasps: Dict[str, Dict] = {}

    object_center = compute_object_center(stage, object_wrapper_path)

    for joint_name, child_link_name, urdf_axis in button_joints:
        print("\n" + "=" * 80)
        print(f"Processing BUTTON joint: {joint_name} (continuous) -> {child_link_name}")
        print(f"  URDF axis: {urdf_axis}")
        print("=" * 80)

        # 1) Sample surfaces on the child link
        child_link_points = collect_all_child_link_surface_samples(
            stage,
            object_ref_path,
            child_link_name,
            n_points_per_mesh=NUM_POINTS_SAMPLED_PER_BUTTON,
        )

        if child_link_points.shape[0] == 0:
            print(f"[ERROR] No surface points sampled for child link {child_link_name}")
            continue

        # 2) Find corresponding USD joint and world parameters
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if joint_prim is None:
            print(f"[ERROR] Could not find USD joint for button {joint_name}")
            continue

        joint_params = get_joint_world_parameters(stage, joint_prim)
        if joint_params is None:
            print(f"[ERROR] Could not extract USD joint parameters for {joint_name}")
            continue

        # Normal = transformed joint axis; contact point = furthest mesh point along that axis
        axis_world = np.asarray(joint_params["axis"], dtype=np.float64)
        button_normal = axis_world  # already unit-length from get_joint_world_parameters

        button_surface_center = get_button_surface_center(child_link_points, button_normal)
        if button_surface_center is None:
            print(f"[WARNING] Could not determine button surface center for {joint_name}")
            continue

        print(f"[DEBUG] Button normal  (joint axis)       : {button_normal}")
        print(f"[DEBUG] Button center  (furthest on axis) : {button_surface_center}")
        debug_visualize_button(
            stage,
            button_surface_center=button_surface_center,
            button_normal=button_normal,
            joint_name=joint_name,
            is_first=(len(all_button_grasps) == 0),  # clear previous debug prims for the first button only
        )
        # 7) Sample a fraction of points closest to button_surface_center for position diversity
        dists = np.linalg.norm(child_link_points - button_surface_center, axis=1)
        n_keep = max(1, int(len(child_link_points) * FRACTION_TO_KEEP_AROUND_BUTTON_CENTER))
        nearest_indices = np.argsort(dists)[:n_keep]
        candidate_positions = child_link_points[nearest_indices]

        print(f"[DEBUG] Using {n_keep}/{len(child_link_points)} surface points near button center for grasp diversity")

        # 8) Generate orientations by rolling around local Z
        grasp_quats = generate_button_grasp_quats(button_normal, num_poses=num_rolls_per_button)
        
        # 9) Pair each quat with a randomly sampled candidate position + gaussian perturbation along local Z
        rng = np.random.default_rng()
        grasp_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        for q in grasp_quats:
            # Pick a random point from the nearest candidates (surface contact point)
            idx = rng.integers(0, len(candidate_positions))
            contact_pos = candidate_positions[idx].copy()

            # Step back from contact point to gripper base frame origin
            gripper_base_pos, _ = offset_pose_along_local_z(
                contact_pos,
                q,
                -GRIPPER_FINGERTIP_OFFSET
            )

            # Slight gaussian perturbation along gripper local Z for depth diversity
            perturb = rng.normal(0.0, 0.005)  # 5mm std
            gripper_base_pos, _ = offset_pose_along_local_z(
                gripper_base_pos,
                q,
                perturb
            )

            grasp_poses.append((gripper_base_pos, q))

        if not grasp_poses:
            print(f"[WARNING] No grasp poses generated for button {joint_name}")
            continue

        all_button_grasps[joint_name] = {
            "grasp_poses": grasp_poses,
            "joint_type": "continuous",
            "child_link_name": child_link_name,
            "joint_axis": axis_world,
            "button_normal": button_normal,
            "button_center": button_surface_center,
        }

        print(f"[SUCCESS] Generated {len(grasp_poses)} button grasp pose(s) for joint {joint_name}")

    print("\n" + "=" * 80)
    print(f"BUTTON SUMMARY: Generated grasps for {len(all_button_grasps)}/{len(button_joints)} continuous joint(s)")
    print("=" * 80 + "\n")

    return all_button_grasps

def evaluate_button_rotation(
    stage,
    joint_prim,
    start_angle_rad: float,
    target_angle_rad: float,
) -> Tuple[bool, float, float]:
    """Evaluate how much a continuous button joint rotated.

    Uses JOINT_SUCCESS_THRESHOLD to decide success based on the
    ratio |delta| / |target|.

    Args:
        stage: USD stage
        joint_prim: USD joint prim for the button
        start_angle_rad: joint angle at the beginning of the trial
        target_angle_rad: desired rotation amount in radians

    Returns:
        success: True if |delta| >= JOINT_SUCCESS_THRESHOLD * |target|
        delta_rad: measured joint angle change
        ratio: |delta| / |target|
    """
    cur = get_joint_current_position(stage, joint_prim)
    if cur is None:
        print("[WARNING] evaluate_button_rotation: could not read joint position")
        return False, 0.0, 0.0

    delta = float(cur - start_angle_rad)
    target_abs = abs(float(target_angle_rad))

    if target_abs < 1e-9:
        print("[WARNING] target_angle_rad is near zero in evaluate_button_rotation")
        return False, delta, 0.0

    delta_abs = abs(delta)
    ratio = delta_abs / target_abs

    threshold = float(JOINT_SUCCESS_THRESHOLD)
    success = ratio >= threshold

    print(
        f"[BUTTON-RESULT] joint={joint_prim.GetPath()} "
        f"delta={np.degrees(delta):.2f}°, "
        f"target={np.degrees(target_angle_rad):.2f}°, "
        f"|delta|/|target|={ratio:.2f}, "
        f"threshold={threshold:.2f}, success={success}"
    )

    return success, delta, ratio

def debug_visualize_button(
    stage,
    button_surface_center: np.ndarray,
    button_normal: np.ndarray,
    joint_name: str,
    arrow_length: float = 0.15,
    is_first: bool = True,
):
    """
    Visualize button geometry using USD prims (visible in Isaac Sim viewport):
      1. RED sphere  = button surface center (contact point)
      2. BLUE line   = normal direction from center (center -> center + normal * arrow_length)

    All prims are written under /World/Debug/Button_{joint_name}/.
    If is_first=True, clears the entire /World/Debug/Button scope first.
    """
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', joint_name)
    debug_root = f"/World/Debug/Button_{safe_name}"

    if is_first:
        scope_root = "/World/Debug"
        old = stage.GetPrimAtPath(scope_root)
        if old.IsValid():
            stage.RemovePrim(scope_root)
        UsdGeom.Scope.Define(stage, scope_root)

    old = stage.GetPrimAtPath(debug_root)
    if old.IsValid():
        stage.RemovePrim(debug_root)
    UsdGeom.Scope.Define(stage, debug_root)

    c = np.asarray(button_surface_center, dtype=np.float64)
    n = np.asarray(button_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    # ------------------------------------------------------------------
    # 1. RED sphere at button surface center
    # ------------------------------------------------------------------
    sphere_path = f"{debug_root}/center_sphere"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.CreateRadiusAttr(0.008)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.1, 0.1)])
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(
        Gf.Vec3d(float(c[0]), float(c[1]), float(c[2]))
    )

    # ------------------------------------------------------------------
    # 2. BLUE line: center -> center + normal * arrow_length
    # ------------------------------------------------------------------
    normal_end = c + n * float(arrow_length)
    normal_line_path = f"{debug_root}/normal_line"
    normal_curve = UsdGeom.BasisCurves.Define(stage, normal_line_path)
    normal_curve.CreateTypeAttr().Set("linear")
    normal_curve.CreatePointsAttr().Set([
        Gf.Vec3f(float(c[0]),          float(c[1]),          float(c[2])),
        Gf.Vec3f(float(normal_end[0]), float(normal_end[1]), float(normal_end[2])),
    ])
    normal_curve.CreateCurveVertexCountsAttr().Set([2])
    normal_curve.CreateWidthsAttr().Set([0.004, 0.004])
    normal_curve.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.5, 1.0)])  # blue

    print(f"[DEBUG-VIZ] Button '{joint_name}':")
    print(f"  center: {c}")
    print(f"  normal: {n}")
    print(f"  normal arrow end: {normal_end}")

#Surface point selection

def normal_to_gripper_quaternion(normal: np.ndarray) -> np.ndarray:
    """
    Convert surface normal to gripper orientation quaternion for push grasping.
    
    The gripper is oriented such that:
    - GRIPPER_APPROACH_LOCAL aligns with the surface normal
    - Gripper rotation around approach axis is randomized for diversity
    
    Args:
        normal: [3] unit normal vector (door outward normal)
    
    Returns:
        [4] quaternion [w, x, y, z] for gripper orientation
    """
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    
    # >>> CHANGED: Use GRIPPER_APPROACH_LOCAL to determine which local axis aligns with normal
    approach_local = np.array(GRIPPER_APPROACH_LOCAL, dtype=np.float64)
    approach_local = approach_local / (np.linalg.norm(approach_local) + 1e-12)
    
    # Target: gripper approach axis should align with normal
    # Use negative normal because gripper approaches opposite to surface normal
    target_world = -normal
    
    # Build rotation that maps approach_local -> target_world
    # We need to find perpendicular vectors to complete the orthonormal basis
    
    # Choose a helper vector that's not parallel to target_world
    if abs(np.dot(target_world, [0, 0, 1])) < 0.9:
        helper = np.array([0, 0, 1], dtype=np.float64)
    else:
        helper = np.array([1, 0, 0], dtype=np.float64)
    
    # Build orthonormal basis with target_world as one axis
    # Determine which local axis (X, Y, or Z) the approach_local represents
    abs_approach = np.abs(approach_local)
    main_axis_idx = np.argmax(abs_approach)
    
    if main_axis_idx == 0:  # approach is along X
        # Build basis: X = target, Y and Z perpendicular
        x_world = target_world
        z_world = np.cross(x_world, helper)
        z_world = z_world / (np.linalg.norm(z_world) + 1e-12)
        y_world = np.cross(z_world, x_world)
        y_world = y_world / (np.linalg.norm(y_world) + 1e-12)
        rotation_matrix = np.column_stack([x_world, y_world, z_world])
    elif main_axis_idx == 1:  # approach is along Y
        # Build basis: Y = target, X and Z perpendicular
        y_world = target_world
        z_world = np.cross(helper, y_world)
        z_world = z_world / (np.linalg.norm(z_world) + 1e-12)
        x_world = np.cross(y_world, z_world)
        x_world = x_world / (np.linalg.norm(x_world) + 1e-12)
        rotation_matrix = np.column_stack([x_world, y_world, z_world])
    else:  # approach is along Z (default case)
        # Build basis: Z = target, X and Y perpendicular
        z_world = target_world
        x_world = np.cross(helper, z_world)
        x_world = x_world / (np.linalg.norm(x_world) + 1e-12)
        y_world = np.cross(z_world, x_world)
        y_world = y_world / (np.linalg.norm(y_world) + 1e-12)
        rotation_matrix = np.column_stack([x_world, y_world, z_world])
    
    # Convert rotation matrix to quaternion
    rot = R.from_matrix(rotation_matrix)
    
    # Add random rotation around approach axis for diversity
    random_angle = np.random.uniform(0, 2 * np.pi)
    rot_random = R.from_rotvec(random_angle * target_world)
    
    # Compose rotations
    rot_final = rot_random * rot
    
    quat_xyzw = rot_final.as_quat()  # scipy format: [x, y, z, w]
    
    # Convert to [w, x, y, z] format
    quat_wxyz = np.array([
        quat_xyzw[3],  # w
        quat_xyzw[0],  # x
        quat_xyzw[1],  # y
        quat_xyzw[2]   # z
    ], dtype=np.float64)
    
    return quat_wxyz
 
# =======================
# Trajectory Planning
# =======================

def get_body_world_position(stage, body_path: str) -> Optional[np.ndarray]:
    """Get world-space position (translation) of a body/link prim, matching the GUI transform."""
    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_xf.ExtractTranslation()

    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)

def get_body_env_local_position(stage, body_path: str, env_root_path: str) -> Optional[np.ndarray]:
    """
    Returns the position of `body_path` in the *environment's local frame*.
    Uses USD xform math directly, avoiding world space altogether.
    """
    body_prim = stage.GetPrimAtPath(body_path)
    env_prim  = stage.GetPrimAtPath(env_root_path)

    if not body_prim.IsValid() or not env_prim.IsValid():
        return None

    body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    env_xf  = UsdGeom.Xformable(env_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Convert body → env-local by removing env's world transform
    env_local_xf = body_xf * env_xf.GetInverse()

    # Extract translation
    t = env_local_xf.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)

def find_usd_joint_prim(stage, object_ref_path: str, joint_name_from_urdf: str) -> Optional[Usd.Prim]:
    """
    Find USD joint prim matching URDF joint name.

    Joints are located at: {object_ref_path}/joints/joint_{num}

    Args:
        stage: USD stage
        object_ref_path: Path to object reference (e.g., /World/Envs/env_0/Object/ref)
        joint_name_from_urdf: Joint name from URDF (e.g., 'joint_0', 'joint_1')

    Returns:
        Joint prim or None
    """
    # Extract joint number from URDF name
    joint_name_clean = joint_name_from_urdf.lower().replace("joint_", "").replace("joint", "")

    asset_root = resolve_asset_root_under_ref(stage, object_ref_path)
    joints_root = f"{asset_root}/joints"
    direct_joint_path = f"{joints_root}/joint_{joint_name_clean}"

    direct_prim = stage.GetPrimAtPath(direct_joint_path)
    if direct_prim.IsValid() and (
        direct_prim.IsA(UsdPhysics.RevoluteJoint) or direct_prim.IsA(UsdPhysics.PrismaticJoint)
    ):
        print(f"[INFO] Found USD joint via direct path: {direct_joint_path}")
        return direct_prim

    # Fallback: search under /joints
    joints_root_prim = stage.GetPrimAtPath(joints_root)
    if not joints_root_prim.IsValid():
        print(f"[ERROR] No /joints folder found at {joints_root}")
        return None

    for prim in joints_root_prim.GetChildren():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue

        prim_name = prim.GetName().lower()

        if (
            joint_name_from_urdf.lower() in prim_name
            or joint_name_clean in prim_name
            or f"joint_{joint_name_clean}" == prim_name
        ):
            print(f"[INFO] Found USD joint via search: {prim.GetPath()}")
            return prim

    print(f"[WARNING] No USD joint found for URDF joint: {joint_name_from_urdf}")
    print(f"[DEBUG] Searched in: {joints_root}")
    print(f"[DEBUG] Available joints under {joints_root}:")
    for prim in joints_root_prim.GetChildren():
        print(f"  - {prim.GetName()} (type: {prim.GetTypeName()})")

    return None


def _gfquat_to_mat3d(q) -> Gf.Matrix3d:
    """Convert pxr Gf.Quat* or 4-seq to a 3x3 rotation matrix."""
    if hasattr(q, "GetReal") and hasattr(q, "GetImaginary"):
        w = float(q.GetReal())
        im = q.GetImaginary()
        x, y, z = float(im[0]), float(im[1]), float(im[2])
    else:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.size != 4:
            raise ValueError(f"Unexpected quat size {arr.size}: {q}")
        # Interpret as [w, x, y, z]
        w, x, y, z = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    rot = R.from_quat([x, y, z, w])  # scipy expects [x, y, z, w]
    m = rot.as_matrix()
    return Gf.Matrix3d(
        m[0, 0],
        m[0, 1],
        m[0, 2],
        m[1, 0],
        m[1, 1],
        m[1, 2],
        m[2, 0],
        m[2, 1],
        m[2, 2],
    )


def _make_transform_gf(t: np.ndarray, q) -> Gf.Matrix4d:
    """Make a Gf.Matrix4d from translation (3,) and quaternion q."""
    t = np.asarray(t, dtype=np.float64).reshape(3)
    r3 = _gfquat_to_mat3d(q)
    m = Gf.Matrix4d(1.0)
    m.SetTranslateOnly(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    # Set rotation block
    m.SetRow3(0, Gf.Vec3d(r3[0][0], r3[0][1], r3[0][2]))
    m.SetRow3(1, Gf.Vec3d(r3[1][0], r3[1][1], r3[1][2]))
    m.SetRow3(2, Gf.Vec3d(r3[2][0], r3[2][1], r3[2][2]))
    return m


def _get_body_world_xf(stage, body_path: str) -> Optional[Gf.Matrix4d]:
    if not body_path:
        return None
    prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return xf


def _compute_joint_frame_world(stage, joint_prim: Usd.Prim) -> Optional[Tuple[Gf.Matrix4d, str]]:
    """Compute joint frame in world using (body0, localPos0/localRot0) if available, else body1."""
    if not joint_prim.IsValid():
        return None

    j = UsdPhysics.Joint(joint_prim)

    # Resolve body0/body1 targets
    body0_targets = []
    body1_targets = []
    try:
        rel0 = j.GetBody0Rel()
        if rel0 and rel0.IsValid():
            body0_targets = rel0.GetTargets()
    except Exception:
        pass
    try:
        rel1 = j.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    body0_path = body0_targets[0].pathString if body0_targets else ""
    body1_path = body1_targets[0].pathString if body1_targets else ""

    # localPos/localRot for each body
    lp0 = (
        j.GetLocalPos0Attr().Get()
        if (j.GetLocalPos0Attr() and j.GetLocalPos0Attr().IsValid())
        else Gf.Vec3f(0, 0, 0)
    )
    lr0 = (
        j.GetLocalRot0Attr().Get()
        if (j.GetLocalRot0Attr() and j.GetLocalRot0Attr().IsValid())
        else Gf.Quatf(1, 0, 0, 0)
    )

    lp1 = (
        j.GetLocalPos1Attr().Get()
        if (j.GetLocalPos1Attr() and j.GetLocalPos1Attr().IsValid())
        else Gf.Vec3f(0, 0, 0)
    )
    lr1 = (
        j.GetLocalRot1Attr().Get()
        if (j.GetLocalRot1Attr() and j.GetLocalRot1Attr().IsValid())
        else Gf.Quatf(1, 0, 0, 0)
    )

    # Prefer body0 frame; fall back to body1
    if body0_path:
        body_xf = _get_body_world_xf(stage, body0_path)
        if body_xf is not None:
            local_xf = _make_transform_gf(np.array([lp0[0], lp0[1], lp0[2]], dtype=np.float64), lr0)
            return body_xf * local_xf, body0_path

    if body1_path:
        body_xf = _get_body_world_xf(stage, body1_path)
        if body_xf is not None:
            local_xf = _make_transform_gf(np.array([lp1[0], lp1[1], lp1[2]], dtype=np.float64), lr1)
            return body_xf * local_xf, body1_path

    return None


def get_joint_world_parameters(stage, joint_prim: Usd.Prim) -> Optional[Dict]:
    """Extract joint parameters in world coordinates.

    IMPORTANT:
      USD Physics joints define their constraint frames via body0/body1 and localPos*/localRot*.
      The joint prim's own Xform is often NOT the physical joint frame.

    Returns:
        Dict with:
        - joint_type: 'revolute' or 'prismatic'
        - axis: [3] world-frame axis
        - pivot: [3] world-frame pivot/reference point (child body center)
        - lower_limit: float (radians for revolute, meters for prismatic)
        - upper_limit: float (radians for revolute, meters for prismatic)
        - limit_units: 'rad', 'deg', or 'm'
        - joint_frame_world: Gf.Matrix4d
        - axis_token: 'X', 'Y', or 'Z'
        - used_body_path: which body was used for pivot
    """
    if not joint_prim.IsValid():
        return None

    # Determine joint type
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        joint_usd = UsdPhysics.RevoluteJoint(joint_prim)
        joint_type = "revolute"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        joint_usd = UsdPhysics.PrismaticJoint(joint_prim)
        joint_type = "prismatic"
    else:
        print(f"[ERROR] Unknown joint type: {joint_prim.GetPath()}")
        return None

    # Compute physical joint frame in world using body0/body1 + localPose
    jf = _compute_joint_frame_world(stage, joint_prim)
    if jf is None:
        print(f"[ERROR] Could not compute joint frame from body0/body1 for {joint_prim.GetPath()}")
        return None
    joint_frame_world, frame_body_path = jf

    # Axis token (X/Y/Z) is defined in the JOINT FRAME
    axis_attr = joint_usd.GetAxisAttr()
    if axis_attr and axis_attr.IsValid():
        axis_token = str(axis_attr.Get())
    else:
        axis_token = "Z"

    axis_map = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    axis_local = axis_map.get(axis_token, axis_map["Z"]).astype(np.float64)

    # Resolve body1 first so we can use its transform for the axis direction.
    body1_targets = []
    try:
        rel1 = joint_usd.GetBody1Rel()
        if rel1 and rel1.IsValid():
            body1_targets = rel1.GetTargets()
    except Exception:
        pass

    body1_path = body1_targets[0].pathString if body1_targets else ""

    # Transform axis using body1's world transform (rotation only).
    # Both the rotation axis and surface normal are expressed as:
    #   origin = body1 world position,  direction = body1_xf * axis_local
    body1_xf = _get_body_world_xf(stage, body1_path) if body1_path else None
    if body1_xf is not None:
        axis_world_gf = body1_xf.TransformDir(
            Gf.Vec3d(float(axis_local[0]), float(axis_local[1]), float(axis_local[2]))
        )
    else:
        # Fallback: use joint frame if body1 is unavailable
        axis_world_gf = joint_frame_world.TransformDir(
            Gf.Vec3d(float(axis_local[0]), float(axis_local[1]), float(axis_local[2]))
        )
    axis_world = np.array(
        [float(axis_world_gf[0]), float(axis_world_gf[1]), float(axis_world_gf[2])],
        dtype=np.float64,
    )
    axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)

    pivot_world = None
    used_body_path = ""

    if body1_path:
        pivot_center = get_body_world_position(stage, body1_path)
        if pivot_center is not None:
            pivot_world = pivot_center
            used_body_path = body1_path

    # Fallback: use the origin of the joint frame if body1 position is unavailable
    if pivot_world is None:
        pivot_world_gf = joint_frame_world.Transform(Gf.Vec3d(0, 0, 0))
        pivot_world = np.array(
            [float(pivot_world_gf[0]), float(pivot_world_gf[1]), float(pivot_world_gf[2])],
            dtype=np.float64,
        )
        used_body_path = frame_body_path

    # Limits
    lower_attr = joint_usd.GetLowerLimitAttr()
    upper_attr = joint_usd.GetUpperLimitAttr()
    lower_raw = lower_attr.Get() if (lower_attr and lower_attr.IsValid()) else None
    upper_raw = upper_attr.Get() if (upper_attr and upper_attr.IsValid()) else None

    if joint_type == "revolute":
        if lower_raw is None or upper_raw is None:
            lower = -np.pi
            upper = np.pi
            limit_units = "rad"
        else:
            lower_f = float(lower_raw)
            upper_f = float(upper_raw)
            # degree-like if clearly > 2*pi
            if max(abs(lower_f), abs(upper_f)) > 6.5:
                lower = np.radians(lower_f)
                upper = np.radians(upper_f)
                limit_units = "deg"
            else:
                lower = lower_f
                upper = upper_f
                limit_units = "rad"
    else:  # prismatic
        lower = float(lower_raw) if lower_raw is not None else -0.5
        upper = float(upper_raw) if upper_raw is not None else 0.5
        limit_units = "m"

    print(f"[DEBUG] Joint frame from {used_body_path}: {joint_prim.GetPath()}")
    print(f"[DEBUG]   axis_token={axis_token}, axis_world={axis_world}, pivot_world={pivot_world}")
    print(f"[DEBUG]   limits=({lower_raw},{upper_raw}) -> ({lower},{upper}) units={limit_units}")

    return {
        "joint_prim": joint_prim,
        "joint_type": joint_type,
        "axis": axis_world,
        "pivot": pivot_world,
        "lower_limit": lower,
        "upper_limit": upper,
        "limit_units": limit_units,
        "joint_frame_world": joint_frame_world,
        "axis_token": axis_token,
        "used_body_path": used_body_path,
    }


def compute_target_joint_displacement(joint_params: Dict) -> float:
    """Compute signed delta motion (angle or distance) from initial position BACKWARD to closed/zero position."""
    lower = float(joint_params["lower_limit"])
    upper = float(joint_params["upper_limit"])
    
    # Compute initial position using the same logic
    initial_pos = compute_initial_joint_position(
        joint_params["joint_type"],
        lower,
        upper
    )
    
    # For backward motion: target is to close (move toward 0)
    # Displacement is NEGATIVE (closing direction)
    CLOSE_SIGN = -1.0
    
    if joint_params["joint_type"] == "revolute" or joint_params["joint_type"] == "continuous":
        # Cap to reasonable closing angle
        target_delta = CLOSE_SIGN * initial_pos * 2
        print(f"[DEBUG] Target revolute delta (backward): {target_delta:.1f}°")
        return float(np.radians(target_delta))
    else:
        # For prismatic: also negative to close
        target_delta = CLOSE_SIGN * initial_pos * 2
        print(f"[DEBUG] Target prismatic delta (backward): {target_delta:.4f}m")
        return float(target_delta)

def compute_gripper_orientation_for_trajectory(
    trajectory: np.ndarray,
    initial_grasp_quat: np.ndarray,
    joint_params: Optional[Dict] = None,
    method: str = "fixed",
) -> np.ndarray:
    """
    Compute gripper orientation at each trajectory waypoint.

    Args:
        trajectory: [num_steps, 3] position waypoints
        initial_grasp_quat: [4] initial quaternion [w, x, y, z]
        joint_params: Dict with joint_type, axis, pivot (needed for revolute)
        method: 'fixed' or 'revolute_follow'

    Returns:
        orientations: [num_steps, 4] quaternions [w, x, y, z]
    """
    num_steps = trajectory.shape[0]

    if method == "fixed":
        return np.tile(initial_grasp_quat, (num_steps, 1))

    elif method == "revolute_follow":
        if joint_params is None or joint_params["joint_type"] != "revolute":
            return np.tile(initial_grasp_quat, (num_steps, 1))
        
        pivot = np.asarray(joint_params["pivot"], dtype=np.float64)
        axis = np.asarray(joint_params["axis"], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        
        # Vector from pivot to initial grasp position
        r0 = trajectory[0] - pivot
        # Project perpendicular to axis (component in rotation plane)
        r0_perp = r0 - np.dot(r0, axis) * axis
        
        orientations = []
        
        for i in range(num_steps):
            # Current position vector from pivot
            r_current = trajectory[i] - pivot
            r_current_perp = r_current - np.dot(r_current, axis) * axis
            
            # Calculate rotation angle from initial to current position
            norm_r0 = np.linalg.norm(r0_perp)
            norm_rc = np.linalg.norm(r_current_perp)
            
            if norm_r0 < 1e-6 or norm_rc < 1e-6:
                # On axis, no rotation
                orientations.append(initial_grasp_quat)
                continue
            
            # Angle between r0_perp and r_current_perp
            cos_theta = np.dot(r0_perp, r_current_perp) / (norm_r0 * norm_rc)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            # Use cross product to determine sign
            cross = np.cross(r0_perp, r_current_perp)
            sin_theta = np.dot(cross, axis) / (norm_r0 * norm_rc)
            
            theta = np.arctan2(sin_theta, cos_theta)
            
            # Create incremental rotation quaternion around axis
            # Rotation axis-angle to quaternion: q = [cos(θ/2), sin(θ/2) * axis]
            half_theta = theta / 2.0
            quat_increment_xyzw = np.array([
                np.sin(half_theta) * axis[0],
                np.sin(half_theta) * axis[1],
                np.sin(half_theta) * axis[2],
                np.cos(half_theta)
            ])
            
            # Convert initial grasp quat from [w,x,y,z] to [x,y,z,w] for scipy
            initial_quat_xyzw = np.array([
                initial_grasp_quat[1],
                initial_grasp_quat[2],
                initial_grasp_quat[3],
                initial_grasp_quat[0]
            ])
            
            # Compose rotations: new_rot = increment_rot * initial_rot
            rot_increment = R.from_quat(quat_increment_xyzw)
            rot_initial = R.from_quat(initial_quat_xyzw)
            rot_new = rot_increment * rot_initial
            
            # Convert back to [w,x,y,z]
            new_quat_xyzw = rot_new.as_quat()
            new_quat_wxyz = np.array([
                new_quat_xyzw[3],  # w
                new_quat_xyzw[0],  # x
                new_quat_xyzw[1],  # y
                new_quat_xyzw[2]   # z
            ])
            
            orientations.append(new_quat_wxyz)
        
        return np.array(orientations)
    
    else:
        raise ValueError(f"Unknown orientation method: {method}")

def plan_revolute_joint_trajectory(
    grasp_position: np.ndarray,
    joint_params: Dict,
    target_angle: float,  # This will be NEGATIVE for closing
    num_steps: int = 200,
) -> np.ndarray:
    """
    Plan circular arc trajectory for revolute joint.
    Supports both opening (positive angle) and CLOSING (negative angle).
    
    For backward motion: target_angle < 0 (moves door toward closed position)
    """
    pivot = joint_params["pivot"]
    axis = joint_params["axis"]

    print(f"[DEBUG] Revolute trajectory (BACKWARD CLOSING):")
    print(f"  Pivot: {pivot}")
    print(f"  Axis: {axis}")
    print(f"  Angle: {np.degrees(target_angle):.1f}° (negative = closing)")

    # Vector from pivot to grasp
    r_vec = grasp_position - pivot
    r_parallel = np.dot(r_vec, axis) * axis
    r_perp = r_vec - r_parallel

    radius = np.linalg.norm(r_perp)
    print(f"  Radius: {radius:.4f}m")

    if radius < 1e-6:
        print(f"[WARNING] Grasp on rotation axis - no circular motion")
        return np.tile(grasp_position, (num_steps, 1))

    # Basis in rotation plane
    e1 = r_perp / radius
    e2 = np.cross(axis, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    trajectory = np.zeros((num_steps, 3), dtype=np.float64)

    # Interpolate from 0 to target_angle (negative for closing)
    for i in range(num_steps):
        theta = (i / max(num_steps - 1, 1)) * target_angle  # theta goes negative
        r_rotated = radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        trajectory[i] = pivot + r_rotated + r_parallel

    return trajectory


def plan_knob_rotation_trajectory(
    grasp_position: np.ndarray,
    grasp_quaternion: np.ndarray,
    joint_params: Dict,
    target_angle: float,
    num_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plan a trajectory for a knob/dial (continuous joint).

    The gripper arcs around the joint axis (position changes) and its
    orientation tracks the rotation around that same axis.

    Returns:
        positions:    [num_steps, 3]
        orientations: [num_steps, 4] quaternions [w, x, y, z]
    """
    pivot = np.asarray(joint_params["pivot"], dtype=np.float64)
    axis  = np.asarray(joint_params["axis"],  dtype=np.float64)
    axis  = axis / (np.linalg.norm(axis) + 1e-12)

    print(f"[DEBUG] Knob rotation trajectory:")
    print(f"  Grasp position: {grasp_position}")
    print(f"  Pivot: {pivot}")
    print(f"  Axis: {axis}")
    print(f"  Angle: {np.degrees(target_angle):.1f}°")

    # --- positions: fixed (no arc movement) ---
    positions = np.zeros((num_steps, 3), dtype=np.float64)
    positions[:] = grasp_position

    # --- orientations: rotate around gripper's local z-axis ---
    initial_quat_xyzw = np.array([
        grasp_quaternion[1], grasp_quaternion[2],
        grasp_quaternion[3], grasp_quaternion[0],
    ])
    rot_initial = R.from_quat(initial_quat_xyzw)
    local_z_world = rot_initial.apply(np.array([0.0, 0.0, 1.0]))

    orientations = np.zeros((num_steps, 4), dtype=np.float64)
    for i in range(num_steps):
        theta = (i / max(num_steps - 1, 1)) * target_angle
        rot_delta = R.from_rotvec(theta * local_z_world)
        rot_new   = rot_delta * rot_initial
        q_xyzw    = rot_new.as_quat()
        orientations[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # [w,x,y,z]

    return positions, orientations


def plan_prismatic_joint_trajectory(
    grasp_position: np.ndarray,
    joint_params: Dict,
    target_distance: float,  # This will be NEGATIVE for closing
    num_steps: int = 200,
) -> np.ndarray:
    """
    Plan linear trajectory for prismatic joint.
    Supports both opening (positive) and CLOSING (negative distance).
    
    For backward motion: target_distance < 0 (moves drawer toward closed)
    """
    axis = joint_params["axis"]

    print(f"[DEBUG] Prismatic trajectory (BACKWARD CLOSING):")
    print(f"  Axis: {axis}")
    print(f"  Distance: {target_distance:.4f}m (negative = closing)")

    trajectory = np.zeros((num_steps, 3), dtype=np.float64)

    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)
        displacement = alpha * target_distance  # Goes negative for closing
        trajectory[i] = grasp_position + displacement * axis

    return trajectory


def offset_pose_along_local_z(
    position: np.ndarray,
    quaternion: np.ndarray,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offset position along gripper's approach direction (GRIPPER_APPROACH_LOCAL).

    Args:
        position: [3] position
        quaternion: [4] quaternion [w,x,y,z]
        offset: Distance (positive = forward, negative = backward)

    Returns:
        (new_position, quaternion)
    """
    # Convert to scipy format [x,y,z,w]
    quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot = R.from_quat(quat_xyzw)

    # >>> CHANGED: Use GRIPPER_APPROACH_LOCAL constant
    approach_local = np.array(GRIPPER_APPROACH_LOCAL, dtype=np.float64)
    approach_world = rot.apply(approach_local)

    new_position = position + approach_world * offset
    return new_position, quaternion


def generate_trajectories_for_all_grasps(
    stage,
    all_grasps: Dict[str, Dict],
    object_ref_path: str,
    num_trajectory_steps: int = 200,
) -> List[Dict]:
    all_trajectories = []

    for joint_name, joint_data in all_grasps.items():
        print(f"\n[INFO] Planning trajectories for: {joint_name}")

        grasp_poses = joint_data["grasp_poses"]
        button_normal = joint_data["button_normal"]
        joint_axis = joint_data["joint_axis"]  # rotation axis in world frame

        # Find joint prim to get pivot point
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if joint_prim is None:
            print(f"[ERROR] Could not find USD joint for {joint_name}")
            continue
        joint_params = get_joint_world_parameters(stage, joint_prim)
        if joint_params is None:
            print(f"[ERROR] Could not extract parameters for {joint_name}")
            continue

        for grasp_idx, (grasp_pos, grasp_quat) in enumerate(grasp_poses):
            try:
                grasp_pos_np = np.asarray(grasp_pos, dtype=np.float64)
                grasp_quat_np = np.asarray(grasp_quat, dtype=np.float64)

                # Arc around joint axis; orientation tracks the joint rotation
                traj_positions, traj_orientations = plan_knob_rotation_trajectory(
                    grasp_pos_np,
                    grasp_quat_np,
                    joint_params,
                    BUTTON_ROTATE_ANGLE,
                    num_trajectory_steps,
                )

                all_trajectories.append({
                    "joint_name": joint_name,
                    "joint_type": "continuous",
                    "grasp_index": grasp_idx,
                    "grasp_position": grasp_pos_np,
                    "grasp_quaternion": grasp_quat_np,
                    "trajectory_positions": traj_positions,
                    "trajectory_orientations": traj_orientations,
                    "target_displacement": BUTTON_ROTATE_ANGLE,
                    "button_normal": button_normal,
                    "joint_pivot_world": joint_params["pivot"],
                })

            except Exception as e:
                print(f"[ERROR] Failed to plan trajectory for grasp {grasp_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[INFO] Generated {len([t for t in all_trajectories if t['joint_name'] == joint_name])} "
              f"trajectories for joint {joint_name}")

    print(f"\n[INFO] Total trajectories: {len(all_trajectories)}")
    return all_trajectories

def create_trajectory_batches(trajectories: List[Dict], num_envs: int) -> List[List[Dict]]:
    """
    Distribute trajectories across environments for batched validation.
    
    Args:
        trajectories: List of all trajectories to validate
        num_envs: Number of cloned environments (NUM_COPIES)
    
    Returns:
        List of batches, where each batch contains trajectories for one round of parallel execution
    """
    batches = []
    
    # Process in chunks of num_envs
    for i in range(0, len(trajectories), num_envs):
        batch = trajectories[i:i + num_envs]
        batches.append(batch)
    
    print(f"[INFO] Split {len(trajectories)} trajectories into {len(batches)} batches")
    print(f"[INFO] Batch sizes: {[len(b) for b in batches]}")
    
    return batches

def create_and_bind_high_friction_material(
    stage, 
    root_prim_path: str,
    static_friction: float = 2.0,
    dynamic_friction: float = 2.0,
    restitution: float = 0.0,
    pre_disabled_instanceable_paths: Optional[List[str]] = None,
):
    """
    Create a high-friction physics material and bind it above instanceable subtrees
    so the binding survives when instanceable is restored later.
    
    Args:
        stage: USD stage
        root_prim_path: Root prim path to apply material to (e.g., OBJECT_REF_PATH)
        static_friction: Static friction coefficient (default: 2.0)
        dynamic_friction: Dynamic friction coefficient (default: 2.0)
        restitution: Bounciness (default: 0.0 for no bounce)
    
    Returns:
        Tuple[int, List[str]]: Number of prims the material was applied to and
        prim paths whose instanceable flag was disabled for authoring.
    """
    # Create material if it doesn't exist
    material_path = "/World/Physics_Materials/HighFrictionMaterial"
    
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
    
    mat_prim = stage.GetPrimAtPath(material_path)
    
    # Apply UsdPhysics.MaterialAPI
    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(mat_prim)
    
    p_mat = UsdPhysics.MaterialAPI(mat_prim)
    p_mat.CreateStaticFrictionAttr().Set(float(static_friction))
    p_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    p_mat.CreateRestitutionAttr().Set(float(restitution))

    # Apply PhysxSchema.PhysxMaterialAPI for combine mode
    if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    
    physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("multiply")
    
    # Get root prim
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        print(f"[ERROR] Invalid root prim path: {root_prim_path}")
        return 0, []
    
    instanceable_disabled_paths: List[str] = []
    candidate_prim_paths: List[str] = []
    bound_target_paths: List[str] = []
    
    # First pass: make the hierarchy editable and collect candidate collision prims.
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        
        # Disable instanceable if set
        if prim.IsInstanceable():
            try:
                prim.SetInstanceable(False)
                instanceable_disabled_paths.append(prim.GetPath().pathString)
            except Exception as e:
                print(f"[WARN] Failed to disable instanceable on {prim.GetPath()}: {e}")
        
        if prim.IsA(UsdGeom.Mesh):
            candidate_prim_paths.append(prim.GetPath().pathString)
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            candidate_prim_paths.append(prim.GetPath().pathString)

    disabled_set = set(instanceable_disabled_paths)
    if pre_disabled_instanceable_paths:
        disabled_set.update(pre_disabled_instanceable_paths)
    bound_target_set = set()

    # Second pass: bind on a stable parent prim instead of deep mesh prims.
    for prim_path in candidate_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        bind_target = prim
        nearest_disabled_ancestor = None
        cursor = prim
        while cursor and cursor.IsValid():
            cursor_path = cursor.GetPath().pathString
            if cursor_path in disabled_set:
                nearest_disabled_ancestor = cursor
                break
            if cursor.GetPath() == root_prim.GetPath():
                break
            cursor = cursor.GetParent()

        if nearest_disabled_ancestor is not None:
            parent = nearest_disabled_ancestor.GetParent()
            if parent and parent.IsValid():
                bind_target = parent
            else:
                bind_target = nearest_disabled_ancestor
        elif prim.IsA(UsdGeom.Mesh):
            parent = prim.GetParent()
            if parent and parent.IsValid():
                bind_target = parent

        bind_target_path = bind_target.GetPath().pathString
        if bind_target_path in bound_target_set:
            continue

        try:
            api = UsdShade.MaterialBindingAPI(bind_target)
            if not api:
                api = UsdShade.MaterialBindingAPI.Apply(bind_target)

            api.Bind(
                UsdShade.Material(mat_prim),
                materialPurpose="physics"
            )
            bound_target_set.add(bind_target_path)
            bound_target_paths.append(bind_target_path)
        except Exception as e:
            print(f"[WARN] Failed to bind material to {bind_target.GetPath()} (from {prim_path}): {e}")
    
    print(f"[INFO] Physics material '{material_path}' applied:")
    print(f"  - Candidate mesh/collision prims: {len(candidate_prim_paths)}")
    print(f"  - Bound to {len(bound_target_paths)} prim(s)")
    print(f"  - Disabled instanceable on {len(instanceable_disabled_paths)} prim(s)")
    print(f"  - Static friction: {static_friction}")
    print(f"  - Dynamic friction: {dynamic_friction}")
    print(f"  - Restitution: {restitution}")
    
    return len(bound_target_paths), instanceable_disabled_paths

def add_lighting(stage):
    light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
    light.CreateIntensityAttr().Set(1000.0)
    d_light = UsdLux.DistantLight.Define(stage, "/World/distantLight")
    d_light.CreateIntensityAttr().Set(2000.0)
    UsdGeom.Xformable(d_light).AddRotateXYZOp().Set(Gf.Vec3f(45, 45, 0))
# =======================
# Processing One Object
# =======================

async def process_one_object(obj_usd: Path, obj_id: str, output_dir: Path, object_wrapper_path: str):
    
    ctx = omni.usd.get_context()
    if ctx.get_stage():
        print(f"[INFO] Closing previous stage...")
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        
        import gc
        gc.collect()
        
        await step_simulation(10)
        
    print(f"[INFO] Creating new stage")
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
        stage.SetDefaultPrim(world)

    if not stage.GetPrimAtPath(ENV_BASE_PATH).IsValid():
        UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    if not stage.GetPrimAtPath(env_path(0)).IsValid():
        UsdGeom.Xform.Define(stage, env_path(0))
    
    add_lighting(stage)
    await omni.kit.app.get_app().next_update_async()
    # Add object to stage
    object_wrapper_xform = UsdGeom.Xform.Define(stage, object_wrapper_path)
    print(f"[INFO] Created wrapper at {object_wrapper_path}")
    
    object_ref_prim = add_reference_to_stage(str(obj_usd), OBJECT_REF_PATH)
    
    if not object_ref_prim:
        print(f"[ERROR] Failed to add object reference")
        return
    print(f"[INFO] Added reference at {OBJECT_REF_PATH}")
    
    gripper_wrapper_xform = UsdGeom.Xform.Define(stage, GRIPPER_WRAPPER_PATH)
    print(f"[INFO]   Created wrapper at {GRIPPER_WRAPPER_PATH}")
    
    gripper_ref_prim = add_reference_to_stage(str(GRIPPER_USD), GRIPPER_REF_PATH)
    if not gripper_ref_prim:
        print(f"[ERROR] Failed to add gripper reference")
        return
    print(f"[INFO]   Added reference at {GRIPPER_REF_PATH}")
    
    await omni.kit.app.get_app().next_update_async()
    
    await omni.kit.app.get_app().next_update_async()
    set_gripper_world_pose(stage, GRIPPER_WRAPPER_PATH, position = [0, 0, 3], quaternion = [1, 0, 0, 0])
    
    await omni.kit.app.get_app().next_update_async()
    
    changed_instanceable_paths = []
    object_material_instanceable_paths = []

    print(f"\n[INFO] Disabling instanceable for grasp generation...")
    changed_instanceable_paths = disable_instanceable_for_grasp_generation(stage, OBJECT_REF_PATH)
    await omni.kit.app.get_app().next_update_async()

    print(f"\n[INFO] Applying high-friction physics material to object...")
    num_prims_with_material, object_material_instanceable_paths = create_and_bind_high_friction_material(
        stage,
        OBJECT_REF_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
        pre_disabled_instanceable_paths=changed_instanceable_paths,
    )

    if num_prims_with_material == 0:
        print(f"[WARN] No prims found to apply physics material")

    await omni.kit.app.get_app().next_update_async()
    
    # Setup physics scene
    print(f"[INFO] Setting up physics scene...")
    physics_scene_path = setup_physics_scene(stage)
    await omni.kit.app.get_app().next_update_async()
    ps_prim = stage.GetPrimAtPath(physics_scene_path)
    if not ps_prim.IsValid():
        raise RuntimeError(f"Failed to create valid physics scene at {physics_scene_path}")
    print(f"[INFO] Physics scene validated at {physics_scene_path}")
    
    # Get bottom center and add ground plane
    bottom_center = get_bbox_bottom_center(stage, object_wrapper_path)
    if bottom_center:
        bottom_center_list = [float(bottom_center[0]), float(bottom_center[1]), float(bottom_center[2])]
        print(f"[INFO] Object bottom center: {bottom_center_list}")
    else:
        bottom_center_list = None
        print(f"[WARN] Could not compute bottom center")
        
    GroundPlane(prim_path="/World/GroundPlane", z_position=bottom_center[2] - 0.0001)
    ground_z = bottom_center[2] - 0.0001
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # SET INITIAL OBJECT POSE
    # =====================
    
    dataset_dir = obj_usd.parent
    urdf_path = os.path.join(str(dataset_dir), 'mobility.urdf')
    joints_info = find_continuous_button_joints_in_urdf(urdf_path)

    print(f"[INFO] Setting initial joint positions before grasp generation...")
    timeline.play()
    await step_simulation(5)

    for joint_name, child_link_name, joint_axis in joints_info:
        joint_prim = find_usd_joint_prim(stage, OBJECT_REF_PATH, joint_name)
        if joint_prim is None:
            continue
        initial_pos = 0.0
        set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, initial_pos, "angular")
        set_joint_position_direct(stage, joint_prim, initial_pos)

    await step_simulation(60)  # Let joints settle

    timeline.pause()
    await step_simulation(5)

    print(f"[INFO] Joints set to initial positions, ready for grasp generation")
    
    # =====================
    # GENERATE GRASPS
    # =====================
    
    print(f"\n[INFO] Generating grasps...")
    
    all_grasps = generate_button_grasps_for_all_joints(
        stage,
        dataset_dir,
        object_wrapper_path,
        OBJECT_REF_PATH,
        GRIPPER_WRAPPER_PATH,
        NUM_POINTS_SAMPLED_PER_BUTTON
    )
    
    if not all_grasps:
        print(f"[ERROR] No grasps generated")
        return
    
    # # =====================
    # # Filtering grasps to avoid overlaps
    # # =====================
    # print(f"\n[INFO] Filtering grasps via overlap detection...")

    # # 1. Disable instanceable on gripper so we can manipulate prims
    # gripper_instanceable_paths = disable_instanceable_for_grasp_generation(stage, GRIPPER_REF_PATH)
    # await omni.kit.app.get_app().next_update_async()

    # # 2. Collect all gripper mesh prims, record which already had CollisionAPI.
    # #    We then DISABLE the collider during the query so the gripper's own
    # #    collision geometry doesn't self-report — only overlaps with the object count.
    # gripper_root_prim = stage.GetPrimAtPath(GRIPPER_REF_PATH)
    # gripper_mesh_paths: List[str] = []          # meshes we touched
    # gripper_had_collision: List[bool] = []       # whether they already had CollisionAPI

    # for prim in Usd.PrimRange(gripper_root_prim):
    #     if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
    #         continue
    #     had = prim.HasAPI(UsdPhysics.CollisionAPI)
    #     gripper_had_collision.append(had)
    #     gripper_mesh_paths.append(prim.GetPath().pathString)

    #     # Add CollisionAPI + convex hull if not already present
    #     if not had:
    #         UsdPhysics.CollisionAPI.Apply(prim)
    #         UsdPhysics.MeshCollisionAPI.Apply(prim)
    #         UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexHull")
        
    #     # Always DISABLE the collider — we re-enable per-mesh only during its query
    #     UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    # print(f"[INFO] Prepared {len(gripper_mesh_paths)} gripper mesh collider(s) (disabled)")
    # await omni.kit.app.get_app().next_update_async()
    
    # physx_iface = get_physx_interface()
    # sq_iface = get_physx_scene_query_interface()

    # def _overlap_at_pose(pos_np: np.ndarray, quat_np: np.ndarray) -> bool:
    #     """
    #     Place gripper at pose, then for each mesh, query
    #     overlap_shape_any. Return True if ANY mesh overlaps.
        
    #     overlap_shape_any returns True if the shape itself has a collider (self-hit)
    #     OR overlaps another collider. By querying each mesh that has collider disabled,
    #     the only possible hits are against other object colliders.
    #     """
    #     set_gripper_world_pose(stage, GRIPPER_WRAPPER_PATH, pos_np, quat_np)
    #     physx_iface.force_load_physics_from_usd()

    #     for mesh_path in gripper_mesh_paths:
    #         mesh_prim = stage.GetPrimAtPath(mesh_path)
    #         if not mesh_prim.IsValid():
    #             continue

    #         enc0, enc1 = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(mesh_path))
    #         hit = sq_iface.overlap_shape_any(enc0, enc1)

    #         if hit:
    #             return True

    #     return False

    # # 3. Filter each joint's grasps
    # filtered_all_grasps: Dict[str, Dict] = {}

    # for joint_name, joint_data in all_grasps.items():
    #     grasp_poses = joint_data["grasp_poses"]
    #     kept = []
    #     rejected = 0

    #     for pos, quat in grasp_poses:
    #         # Normalise types
    #         if isinstance(pos, (Gf.Vec3d, Gf.Vec3f)):
    #             pos_np = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
    #         else:
    #             pos_np = np.asarray(pos, dtype=np.float64)

    #         if isinstance(quat, (Gf.Quatd, Gf.Quatf)):
    #             w = float(quat.GetReal()); im = quat.GetImaginary()
    #             quat_np = np.array([w, float(im[0]), float(im[1]), float(im[2])], dtype=np.float64)
    #         else:
    #             quat_np = np.asarray(quat, dtype=np.float64)

    #         if _overlap_at_pose(pos_np, quat_np):
    #             rejected += 1
    #             await step_simulation(100)
    #         else:
    #             kept.append((pos, quat))

    #     print(f"[INFO] Joint '{joint_name}': {len(grasp_poses)} → {len(kept)} "
    #           f"(rejected {rejected} overlapping grasps)")

    #     if kept:
    #         filtered_all_grasps[joint_name] = {**joint_data, "grasp_poses": kept}

    # all_grasps = filtered_all_grasps
    # # 4. Restore gripper colliders to their original state
    # for mesh_path, had_collision in zip(gripper_mesh_paths, gripper_had_collision):
    #     prim = stage.GetPrimAtPath(mesh_path)
    #     if not prim.IsValid():
    #         continue
    #     if had_collision:
    #         # It had a collider before — re-enable it
    #         UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
    #     else:
    #         # We added the collider — remove it entirely
    #         prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    #         prim.RemoveAPI(UsdPhysics.CollisionAPI)

    # print(f"[INFO] Restored gripper collider state on {len(gripper_mesh_paths)} mesh(es)")

    # # 5. Restore gripper instanceable state
    # restore_instanceable(stage, gripper_instanceable_paths)
    # await omni.kit.app.get_app().next_update_async()
    
    # # 6. Final PhysX reload to evict any temporary colliders from the simulation
    # physx_iface.force_load_physics_from_usd()
    # await omni.kit.app.get_app().next_update_async()

    # print(f"[INFO] Overlap filtering complete. "
    #       f"Joints with valid grasps: {len(all_grasps)}")
    
    # Keep the asset-authored rigid-body/gravity settings for rotate tasks.
    # Reapplying object-wide PhysX overrides made button contacts unstable.
    # =====================
    # TRAJECTORY GENERATION 
    # =====================
    print("[INFO] Generating trajectories for all grasps...")
    all_trajectories = generate_trajectories_for_all_grasps(
        stage,
        all_grasps,
        OBJECT_REF_PATH,
        num_trajectory_steps=200,
    )
    
    # =====================
    # Restore joint angle
    # =====================
    print(f"\n[INFO] Restoring joint initial positions after grasp generation...")
    
    print(f"[INFO] Setting initial joint positions after grasp generation...")
    timeline.play()
    await step_simulation(5)
    
    for joint_name, child_link_name, joint_axis in joints_info:
        joint_prim = find_usd_joint_prim(stage, OBJECT_REF_PATH, joint_name)
        if joint_prim is None:
            continue
        initial_pos = 0.0
        set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, initial_pos, "angular")
        set_joint_position_direct(stage, joint_prim, initial_pos)

    await step_simulation(60)  # Let joints settle
    timeline.stop()
    print(f"[INFO] Joints set to initial positions, ready for grasp generation")
    
    # =====================
    # RESTORE INSTANCEABLE
    # =====================
    print(f"\n[INFO] Restoring instanceable state...")
    restore_instanceable(stage, object_material_instanceable_paths + changed_instanceable_paths)
    await omni.kit.app.get_app().next_update_async()
    
    # =====================
    # GRID CLONER
    # =====================
    print(f"[INFO] Cloning template env_0 into grid (NUM_COPIES={NUM_COPIES}, spacing={CLONE_SPACING})...")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_COPIES)
    print(f"[DEBUG] Cloner env_paths[0..min]: {env_paths[:min(5, len(env_paths))]}")
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths)
    await omni.kit.app.get_app().next_update_async()
    
    await step_simulation(2)  # Let clones initialize
    # Debug: confirm env transforms differ (so clones are spatially separated)
    try:
        p0, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(0)))
        p1, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(1))) if NUM_COPIES > 1 else (None, None)
        print(f"[DEBUG] env_0 world pos: {p0}")
        if p1 is not None:
            print(f"[DEBUG] env_1 world pos: {p1} (delta ~ {p1 - p0})")
    except Exception as e:
        print(f"[WARN] Could not print env transforms: {e}")
    
    print(f"[INFO] Cloning complete.")
    
    # Fix object base to world in all envs
    print(f"\n[INFO] Adding fixed joints to all {NUM_COPIES} environments...")
    for i in range(NUM_COPIES):
        env_obj_ref = obj_ref(i)
        fixed_joint_path = f"{env_path(i)}/ObjectFixedToWorld"
        env_root_path = resolve_asset_root_under_ref(stage, env_obj_ref)
        print(f"[INFO] Fixing object in env_{i}...")
        fix_object_base_to_world(
            stage, 
            env_root_path, 
            base_link_name="base", 
            joint_path=fixed_joint_path
        )
    
    await omni.kit.app.get_app().next_update_async()
    print(f"[INFO] All environments fixed to world")
    await step_simulation(2)

    # =====================
    # PHYSICS VALIDATION
    # =====================
    valid_trajectories = []
    if all_trajectories:
        print("[INFO] Running physics validation on trajectories (step-back + approach)...")
        valid_trajectories = await physics_validation_loop(
            stage,
            all_trajectories,
            ground_z,
        )
        print(
            f"[INFO] Physics validation kept {len(valid_trajectories)}/"
            f"{len(all_trajectories)} trajectories"
        )
        
        # Save validated trajectories to JSON
        if valid_trajectories and bottom_center_list:
            try:
                json_path = save_trajectories_to_json(
                    valid_trajectories,
                    obj_usd,
                    bottom_center_list
                )
                print(f"[SUCCESS] Trajectory data saved to {json_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save trajectory JSON: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("[WARN] No trajectories generated; skipping physics validation")
    
    # Cleanup
        
    if PROCESSING_MODE == "dataset":
        mark_object_completed(LOG_FILE, obj_id)
    
    print(f"\n{'='*80}")
    print(f"Completed: {obj_id}")
    print(f"{'='*80}\n")

# =======================
# Main Pipeline Functions
# =======================

async def run_pipeline():
    """Main pipeline - processes either single object or entire dataset"""
    
    print(f"\n{'#'*80}")
    print(f"# Grasp Generation and Validation Pipeline")
    print(f"# Mode: {PROCESSING_MODE.upper()}")
    print(f"{'#'*80}\n")
    
    if PROCESSING_MODE == "single":
        # =====================
        # SINGLE OBJECT MODE
        # =====================
        print(f"[INFO] Processing single object: {SINGLE_OBJECT_USD}")
        
        if not SINGLE_OBJECT_USD.exists():
            print(f"[ERROR] Object file not found: {SINGLE_OBJECT_USD}")
            return
        
        obj_id = SINGLE_OBJECT_USD.parent.name
        output_dir = SINGLE_OBJECT_USD.parent
        
        try:
            await process_one_object(SINGLE_OBJECT_USD, obj_id, output_dir, OBJECT_WRAPPER_PATH)
            print(f"\n[SUCCESS] Single object processing complete!")
        except Exception as e:
            print(f"\n[ERROR] Failed to process object: {e}")
            import traceback
            traceback.print_exc()
    
    elif PROCESSING_MODE == "dataset":
        # =====================
        # DATASET MODE
        # =====================
        print(f"[INFO] Dataset path: {INPUT_DATASET_PATH}")
        print(f"[INFO] Log file: {LOG_FILE}")
        
        if not INPUT_DATASET_PATH.exists():
            print(f"[ERROR] Dataset path not found: {INPUT_DATASET_PATH}")
            return
        
        # Find all objects
        all_objects = find_all_objects(INPUT_DATASET_PATH)
        print(f"[INFO] Found {len(all_objects)} objects in dataset")
        
        if not all_objects:
            print(f"[ERROR] No objects found in {INPUT_DATASET_PATH}")
            return
        
        # Get completed objects
        completed = get_completed_objects(LOG_FILE)
        print(f"[INFO] Already completed: {len(completed)} objects")
        
        # Filter out completed objects
        remaining = [(usd, oid) for usd, oid in all_objects if oid not in completed]
        print(f"[INFO] Remaining to process: {len(remaining)} objects")
        
        if not remaining:
            print(f"[INFO] All objects already processed!")
            return
        
        # Process each object
        success_count = 0
        fail_count = 0
        
        for idx, (obj_usd, obj_id) in enumerate(remaining, start=1):
            print(f"\n{'='*80}")
            print(f"Processing {idx}/{len(remaining)}: {obj_id}")
            print(f"{'='*80}\n")
            
            try:
                output_dir = obj_usd.parent
                await process_one_object(obj_usd, obj_id, output_dir, OBJECT_WRAPPER_PATH)
                success_count += 1
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process {obj_id}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
                print(f"[INFO] Continuing to next object...")
                continue
        
        print(f"\n{'#'*80}")
        print(f"# Dataset Processing Complete")
        print(f"# Successfully processed: {success_count}/{len(remaining)}")
        print(f"# Failed: {fail_count}/{len(remaining)}")
        print(f"{'#'*80}\n")
    
    else:
        print(f"[ERROR] Invalid PROCESSING_MODE: '{PROCESSING_MODE}'")
        print(f"[ERROR] Must be either 'single' or 'dataset'")
        return
    
    print(f"\n{'#'*80}")
    print(f"# Pipeline Complete")
    print(f"{'#'*80}\n")

def main():
    """Entry point"""
    print("[INFO] Starting pipeline...")
    
    try:
        task = asyncio.ensure_future(run_pipeline())
        
        while not task.done():
            simulation_app.update()
        
        if task.exception():
            raise task.exception()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Closing simulation app...")
        simulation_app.close()
        print("[INFO] Done")


# =======================
# Physics validation loop
# =======================

# Helper: Set USD joint drive target (for gripper closing etc.)
def set_usd_joint_drive_target(stage, joint_prim_path: str, target_value: float, drive_kind: str = "linear"):
    """Set a USD Physics drive target on a joint prim.

    Used for closing the gripper during physics validation.

    Args:
        stage: USD stage
        joint_prim_path: Path to a USD joint prim (e.g., .../panda_finger_joint1)
        target_value: target position (meters for prismatic, radians for revolute)
        drive_kind: "linear" for prismatic joints, "angular" for revolute joints
    """
    prim = stage.GetPrimAtPath(joint_prim_path)
    if not prim.IsValid():
        print(f"[WARN] set_usd_joint_drive_target: invalid joint prim {joint_prim_path}")
        return

    try:
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_kind)

        tp = drive.GetTargetPositionAttr()
        if not tp or not tp.IsValid():
            tp = drive.CreateTargetPositionAttr()
        tp.Set(float(target_value))

        # If the asset already has gains, these may already be set.
        # We set defaults only when they are missing.
        st = drive.GetStiffnessAttr()
        if not st or not st.IsValid():
            st = drive.CreateStiffnessAttr()
        if st.Get() is None:
            st.Set(400.0)

        dm = drive.GetDampingAttr()
        if not dm or not dm.IsValid():
            dm = drive.CreateDampingAttr()
        if dm.Get() is None:
            dm.Set(80.0)

        mf = drive.GetMaxForceAttr()
        if not mf or not mf.IsValid():
            mf = drive.CreateMaxForceAttr()
        if mf.Get() is None:
            mf.Set(20.0)

    except Exception as e:
        print(f"[WARN] Failed to set drive target on {joint_prim_path} ({drive_kind}): {e}")

def compute_initial_joint_position(joint_type: str, lower_limit: float, upper_limit: float) -> float:

    if joint_type == "revolute":
        # Convert constant from degrees to radians
        abs_value = np.radians(INITIAL_JOINT_ANGLE)
        
        # Compute ratio-based value
        joint_range = upper_limit - lower_limit
        ratio_value = lower_limit + INITIAL_JOINT_ANGLE_RATIO * joint_range
        
        # Take minimum (clamped to joint limits)
        initial_pos = min(abs_value, ratio_value)
        initial_pos = np.clip(initial_pos, lower_limit, upper_limit)
        
        print(f"[DEBUG] Revolute: abs={np.degrees(abs_value)}°, "
              f"ratio={np.degrees(ratio_value):.1f}° (range=[{np.degrees(lower_limit):.1f}°, {np.degrees(upper_limit):.1f}°]), "
              f"using {np.degrees(initial_pos):.1f}°")
        
        return float(np.degrees(initial_pos))
        
    elif joint_type == "prismatic":  # prismatic
        # Absolute value in meters
        abs_value = INITIAL_JOINT_POSITION_M
        
        # Compute ratio-based value
        joint_range = upper_limit - lower_limit
        ratio_value = lower_limit + INITIAL_JOINT_POSITION_M_RATIO * joint_range
        
        # Take minimum (clamped to joint limits)
        initial_pos = min(abs_value, ratio_value)
        initial_pos = np.clip(initial_pos, lower_limit, upper_limit)
        
        print(f"[DEBUG] Prismatic: abs={abs_value:.3f}m, "
              f"ratio={ratio_value:.3f}m (range=[{lower_limit:.3f}m, {upper_limit:.3f}m]), "
              f"using {initial_pos:.3f}m")
    
        return float(initial_pos)

def set_joint_position_direct(stage, joint_prim: Usd.Prim, position: float):
    """
    Directly set joint position using the Joint State API.
    This corresponds to GUI: Joint State -> Angular/Linear -> Position
    
    Args:
        stage: USD stage
        joint_prim: Joint prim (RevoluteJoint or PrismaticJoint)
        position: Target position (RADIANS for revolute, meters for prismatic)
        also_set_drive: If True, also set drive target to match (keeps drive consistent)
    """
    if not joint_prim.IsValid():
        print(f"[WARN] set_joint_position_direct: invalid joint prim")
        return
    
    joint_type = None
    drive_kind = None
    state_api_kind = None
    
    # Determine joint type, drive kind, and state API kind
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        joint_type = "revolute"
        drive_kind = "angular"
        state_api_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        joint_type = "prismatic"
        drive_kind = "linear"
        state_api_kind = "linear"
    else:
        print(f"[WARN] set_joint_position_direct: unknown joint type {joint_prim.GetTypeName()}")
        return
    
    # Set the joint state position using PhysxSchema.JointStateAPI
    try:
        # Apply JointStateAPI if not already present
        if not joint_prim.HasAPI(PhysxSchema.JointStateAPI):
            PhysxSchema.JointStateAPI.Apply(joint_prim, state_api_kind)
        
        joint_state = PhysxSchema.JointStateAPI(joint_prim, state_api_kind)
        
        # Set the position attribute
        pos_attr = joint_state.GetPositionAttr()
        if not pos_attr or not pos_attr.IsValid():
            pos_attr = joint_state.CreatePositionAttr()
        pos_attr.Set(float(position))
        
        print(f"[DEBUG] Set joint state position: {joint_prim.GetPath()} = {position:.4f} "
              f"({'deg' if joint_type == 'revolute' else 'm' if state_api_kind == 'linear' else 'rad'})")
    
    except Exception as e:
        print(f"[ERROR] Failed to set joint state position on {joint_prim.GetPath()}: {e}")
        import traceback
        traceback.print_exc()
        
def set_gripper_world_pose(stage, gripper_wrapper_path: str, position: np.ndarray, quaternion: np.ndarray):
    """
    Set the gripper wrapper prim's world pose using a translate + orient op.

    Args:
        stage: USD stage
        gripper_wrapper_path: Path to the gripper wrapper Xform (e.g., GRIPPER_WRAPPER_PATH)
        position: [3] world position in meters
        quaternion: [4] quaternion [w, x, y, z]
    """
    prim = stage.GetPrimAtPath(gripper_wrapper_path)
    if not prim.IsValid():
        print(f"[WARN] set_gripper_world_pose: invalid prim {gripper_wrapper_path}")
        return

    xformable = UsdGeom.Xformable(prim)
    ops = list(xformable.GetOrderedXformOps())

    translate_op = None
    orient_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    wxyz = np.asarray(quaternion, dtype=np.float64).reshape(4)
    w, x, y, z = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])

    translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

async def validate_batch_parallel(
    stage,
    batch: List[Dict],
    batch_index: int,
    ground_z: float,
) -> List[Tuple[int, bool, Optional[int]]]:
    """Validate a batch of trajectories in parallel across environments."""
    print(f"\n[BATCH {batch_index}] Validating {len(batch)} trajectories in parallel...")
    
    if timeline.is_playing() or timeline.is_stopped() == False:
        timeline.stop()
        await step_simulation(10)
        
    # Step 0: Reset gripper positions to avoid physics instability
    for env_idx in range(len(batch)):
        gripper_wrapper_path = grip_wrap(env_idx)
        hand_prim = stage.GetPrimAtPath(grip_base(env_idx))
        hand_prim.GetAttribute("physics:velocity").Set((0, 0, 0))
        hand_prim.GetAttribute("physics:angularVelocity").Set((0, 0, 0))
        set_gripper_world_pose(stage, gripper_wrapper_path, position=[0, 0, 3], quaternion=[1, 0, 0, 0])
    
    await step_simulation(2)

    # Step 1: Reset all joints to 0
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_ref_path = obj_ref(env_idx)
        joint_name = traj.get('joint_name', '')
        print(f"[BATCH {batch_index}][ENV {env_idx}] Resetting joint '{joint_name}' to 0...")
        
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        if joint_prim is None:
            print(f"[WARN] Env {env_idx}: Could not find joint '{joint_name}', skipping reset")
            continue
        
        # Continuous joints always reset to 0
        set_usd_joint_drive_target(stage, joint_prim.GetPath().pathString, 0.0, "angular")
        set_joint_position_direct(stage, joint_prim, 0.0)
    
    await step_simulation(2)
    await ensure_timeline_playing()
    await step_simulation(60)  # Let resets settle
    timeline.pause()
    await step_simulation(2)

    # Step 2: Record initial states for all envs
    initial_states = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        object_ref_path = obj_ref(env_idx)
        joint_name = traj.get('joint_name', '')
        
        joint_prim = find_usd_joint_prim(stage, object_ref_path, joint_name)
        initial_joint_pos = get_joint_current_position(stage, joint_prim) if joint_prim else 0.0
        
        traj['initial_joint_pos'] = initial_joint_pos
        initial_states.append({
            'initial_joint_pos': initial_joint_pos,
            'joint_prim': joint_prim,
        })
    
    # Step 3: Compute stepback positions (for ground z check only)
    stepback_positions = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        grasp_pos_np = np.asarray(traj["grasp_position"], dtype=np.float64).reshape(3)
        grasp_quat_np = np.asarray(traj["grasp_quaternion"], dtype=np.float64).reshape(4)
        stepback_pos, _ = offset_pose_along_local_z(grasp_pos_np, grasp_quat_np, -APPROACH_DISTANCE)
        stepback_positions.append(stepback_pos)

    env_failed = [False] * len(batch)

    # Step 4: Teleport gripper directly to grasp pose (hard set, no approach motion)
    open_target = 0.04

    if timeline.is_playing():
        timeline.stop()
        await step_simulation(2)

    for env_idx in range(len(batch)):
        if env_failed[env_idx]:
            continue
        gripper_wrapper_path = grip_wrap(env_idx)
        stepback_pos = stepback_positions[env_idx]
        grasp_pos_np = np.asarray(batch[env_idx]["grasp_position"], dtype=np.float64)
        grasp_quat_np = np.asarray(batch[env_idx]["grasp_quaternion"], dtype=np.float64)
        gripper_ref_path = grip_ref(env_idx)

        if stepback_pos[2] < ground_z:
            env_failed[env_idx] = True
            print(f"[REJECT] Env {env_idx}: Grasp z={grasp_pos_np[2]:.4f} below ground z={ground_z:.4f}")
            continue

        finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
        finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
        set_usd_joint_drive_target(stage, finger_joint1, open_target, "linear")
        set_usd_joint_drive_target(stage, finger_joint2, open_target, "linear")

        set_gripper_world_pose(stage, gripper_wrapper_path, grasp_pos_np, grasp_quat_np)

    await step_simulation(5)

    if all(env_failed):
        print(f"[BATCH {batch_index}] All environments failed at placement.")
        return [(traj.get('original_index', -1), False, None) for traj in batch]

    # Step 5: Close gripper
    timeline.stop()
    close_target = 0.0
    for env_idx in range(len(batch)):
        if env_failed[env_idx]:
            continue
        gripper_wrapper_path = grip_wrap(env_idx)
        gripper_ref_path = grip_ref(env_idx)
        
        grasp_pos = np.asarray(batch[env_idx]["grasp_position"], dtype=np.float64)
        grasp_quat = np.asarray(batch[env_idx]["grasp_quaternion"], dtype=np.float64)
        
        set_gripper_world_pose(stage, gripper_wrapper_path, grasp_pos, grasp_quat)
        
        finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
        finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
        set_usd_joint_drive_target(stage, finger_joint1, close_target, "linear")
        set_usd_joint_drive_target(stage, finger_joint2, close_target, "linear")
        
    await step_simulation(5)
    await ensure_timeline_playing()
    
    for _ in range(CLOSE_STEPS):
        for env_idx in range(len(batch)):
            if env_failed[env_idx]:
                continue
            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
        await step_simulation(1)

    if all(env_failed):
        print(f"[BATCH {batch_index}] All environments failed after close phase.")
        return [(traj.get('original_index', -1), False, None) for traj in batch]

    # Step 6: Hold phase
    for _ in range(HOLD_STEPS):
        for env_idx in range(len(batch)):
            if env_failed[env_idx]:
                continue
            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
        await step_simulation(1)

    # Step 7: Execute trajectory phase
    cached_traj_pos = []
    cached_traj_ori = []
    cached_traj_len = []
    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        tp = np.asarray(traj.get("trajectory_positions", []), dtype=np.float64)
        to = np.asarray(traj.get("trajectory_orientations", []), dtype=np.float64)
        cached_traj_pos.append(tp)
        cached_traj_ori.append(to)
        cached_traj_len.append(tp.shape[0])

    max_traj_length = max(cached_traj_len) if cached_traj_len else 0

    env_finished_early = [False] * len(batch)
    env_actual_steps = [0] * len(batch)
    env_freeze_pos = [None] * len(batch)
    env_freeze_quat = [None] * len(batch)

    FREEZE_THRESHOLD_RAD = np.radians(0.5)  # within 0.5° of target = done

    for traj_step in range(max_traj_length):
        for env_idx in range(len(batch)):
            if env_failed[env_idx]:
                continue
            
            if env_finished_early[env_idx]:
                # Hold frozen pose
                gripper_wrapper_path = grip_wrap(env_idx)
                set_gripper_world_pose(stage, gripper_wrapper_path, env_freeze_pos[env_idx], env_freeze_quat[env_idx])
                gripper_ref_path = grip_ref(env_idx)
                finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
                finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
                set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
                set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")
                continue

            if traj_step >= cached_traj_len[env_idx]:
                continue

            pos = cached_traj_pos[env_idx][traj_step]
            quat = cached_traj_ori[env_idx][traj_step]

            gripper_wrapper_path = grip_wrap(env_idx)
            set_gripper_world_pose(stage, gripper_wrapper_path, pos, quat)

            gripper_ref_path = grip_ref(env_idx)
            finger_joint1 = f"{gripper_ref_path}/panda_hand/panda_finger_joint1"
            finger_joint2 = f"{gripper_ref_path}/panda_hand/panda_finger_joint2"
            set_usd_joint_drive_target(stage, finger_joint1, close_target, drive_kind="linear")
            set_usd_joint_drive_target(stage, finger_joint2, close_target, drive_kind="linear")

        await step_simulation(TRAJECTORY_SIM_STEPS_PER_WAYPOINT)

        # Check for sufficient rotation — early finish
        for env_idx in range(len(batch)):
            if env_failed[env_idx] or env_finished_early[env_idx]:
                continue
            
            local_len = cached_traj_len[env_idx]
            if traj_step >= local_len:
                continue
            
            joint_prim = initial_states[env_idx]['joint_prim']
            current_joint_pos = get_joint_current_position(stage, joint_prim)
            if current_joint_pos is None:
                continue

            initial_pos = initial_states[env_idx]['initial_joint_pos']
            target_disp = batch[env_idx].get("target_displacement", 0.0)
            rotated = current_joint_pos - initial_pos  # positive = correct direction

            # Within FREEZE_THRESHOLD_RAD of target AND moving in right direction
            remaining = abs(target_disp) - rotated
            if rotated > 0 and remaining < FREEZE_THRESHOLD_RAD:
                env_finished_early[env_idx] = True
                env_actual_steps[env_idx] = min(traj_step + 2, local_len - 1)
                env_freeze_pos[env_idx] = cached_traj_pos[env_idx][traj_step].copy()
                env_freeze_quat[env_idx] = cached_traj_ori[env_idx][traj_step].copy()
                print(f"[BATCH {batch_index}] Env {env_idx}: Early finish at step {traj_step}, "
                      f"rotated={np.degrees(rotated):.1f}°")

    # Step 8: Determine success
    results = []

    for env_idx in range(len(batch)):
        traj = batch[env_idx]
        original_idx = traj.get('original_index', -1)
        
        if env_failed[env_idx]:
            results.append((original_idx, False, None))
            continue
        
        if env_finished_early[env_idx]:
            termination_step = env_actual_steps[env_idx]
            print(f"[BATCH {batch_index}] Env {env_idx}: SUCCESS (early termination at step {termination_step})")
            results.append((original_idx, True, termination_step))
            continue
        
        joint_prim = initial_states[env_idx]['joint_prim']
        initial_joint_pos = initial_states[env_idx]['initial_joint_pos']
        target_displacement = traj.get("target_displacement", 0.0)
        
        final_joint_pos = get_joint_current_position(stage, joint_prim)
        if final_joint_pos is None:
            results.append((original_idx, False, None))
            continue
        
        actual_displacement = final_joint_pos - initial_joint_pos  # signed
        required_displacement = JOINT_SUCCESS_THRESHOLD * abs(target_displacement)
        
        # Must have rotated enough AND in the positive direction
        is_valid = (actual_displacement >= required_displacement)
        
        print(f"[BATCH {batch_index}] Env {env_idx}: "
              f"Initial={np.degrees(initial_joint_pos):.1f}°, "
              f"Final={np.degrees(final_joint_pos):.1f}°, "
              f"Rotated={np.degrees(actual_displacement):.1f}°, "
              f"Required={np.degrees(required_displacement):.1f}°, "
              f"Valid={is_valid}")
        
        results.append((original_idx, is_valid, None))

    return results

def get_joint_current_position(stage, joint_prim: Usd.Prim) -> Optional[float]:
    """Read current joint position by checking the computed joint value attribute.
    
    Returns:
        Current position (radians for revolute, meters for prismatic)
    """
    if not joint_prim.IsValid():
        return None
    
    try:
        
        # Check if joint has PhysX joint state API
        if joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
            physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
            
            # Try to get the joint position from PhysX schema
            # The actual position might be stored differently
            pass
        
        # Get DC interface
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        
        # Get articulation
        articulation = dc.get_articulation(joint_prim.GetPath().pathString)
        if articulation == _dynamic_control.INVALID_HANDLE:
            # Try parent
            parent = joint_prim.GetParent()
            while parent and parent.IsValid():
                articulation = dc.get_articulation(parent.GetPath().pathString)
                if articulation != _dynamic_control.INVALID_HANDLE:
                    break
                parent = parent.GetParent()
        
        if articulation == _dynamic_control.INVALID_HANDLE:
            print(f"[WARN] Could not get articulation handle")
            return None
        
        # Get joint handle
        dof_ptr = dc.find_articulation_dof(articulation, joint_prim.GetName())
        
        if dof_ptr != _dynamic_control.INVALID_HANDLE:
            # Read DOF position
            dof_state = dc.get_dof_state(dof_ptr, _dynamic_control.STATE_POS)
            position = dof_state.pos
            return float(position)
        
        print(f"[WARN] Could not get DOF handle for joint {joint_prim.GetName()}")
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to read joint position: {e}")
        import traceback
        traceback.print_exc()
        return None

async def physics_validation_loop(
    stage,
    trajectories: List[Dict],
    ground_z: float,
) -> List[Dict]:
    """
    Run batched parallel physics validation across all cloned environments.
    """
    if not trajectories:
        return []
    
    # Add original index to each trajectory for tracking
    for idx, traj in enumerate(trajectories):
        traj['original_index'] = idx
    
    # Create batches
    batches = create_trajectory_batches(trajectories, NUM_COPIES)
    
    # >>> CHANGED: Track both success and termination_step
    all_results = {}  # {original_index: (success_bool, termination_step)}
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        results = await validate_batch_parallel(stage, batch, batch_idx, ground_z)
        
        # >>> CHANGED: Store both success and termination step
        for orig_idx, success, termination_step in results:
            all_results[orig_idx] = (success, termination_step)
    
    # Filter to valid trajectories and store termination step
    valid = []
    for traj in trajectories:
        orig_idx = traj['original_index']
        if orig_idx in all_results:
            success, termination_step = all_results[orig_idx]
            if success:
                # >>> NEW: Store termination step if early termination occurred
                if termination_step is not None:
                    traj['termination_step'] = termination_step
                valid.append(traj)
    
    print(f"\n[INFO] Batched validation: {len(valid)}/{len(trajectories)} trajectories passed")
    return valid

if __name__ == "__main__":
    main()
