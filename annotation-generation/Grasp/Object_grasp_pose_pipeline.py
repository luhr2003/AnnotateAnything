from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import asyncio
import os
import json
import yaml
import numpy as np
import math
from pathlib import Path
from collections import Counter, defaultdict
import sys
import io
from contextlib import contextmanager
import shutil

# Isaac Sim imports
import omni.kit.app
import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
from omni.physx.scripts import physicsUtils
from isaacsim.core.cloner import GridCloner 
from isaacsim.core.utils.stage import get_current_stage, get_current_stage_id
from isaacsim.core.simulation_manager import SimulationManager
import omni.physics.tensors
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
    ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
timeline = get_timeline_interface()
from isaacsim.replicator.grasping.grasping_manager import GraspingManager
import isaacsim.replicator.grasping.grasping_utils as grasping_utils
import isaacsim.replicator.grasping.transform_utils as transform_utils

# =======================
# CONFIG
# =======================
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

INPUT_ROOT = _path_from_env("GRASP_INPUT_ROOT", _THIS_DIR)
GRIPPER_USD = _path_from_env("GRASP_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")
DATASET_PATH = _path_from_env("GRASP_DATASET_PATH", _THIS_DIR)
FUNCTIONAL_PAIRS_PATH = _path_from_env("GRASP_FUNCTIONAL_PAIRS_PATH", _THIS_DIR / "functional_list.json")
TARGET_OUTPUT_ROOT = INPUT_ROOT
LOG_FILE = TARGET_OUTPUT_ROOT / "completed_objects.txt"

# Processing control flags
BYPASS_RECENTER_AND_FLATTEN = True
HAS_PARTS = False
CLEAN_EXISTING_RIGID_BODY = True

PHYSICS_SCENE_PATH = "/World/physicsScene"

# =======================
# PHYSICS MATERIAL / FRICTION
# =======================
APPLY_HIGH_FRICTION_MATERIAL = True
HIGH_FRICTION_MATERIAL_PATH = "/World/PhysicsMaterials/HighFriction"

HIGH_STATIC_FRICTION = 2
HIGH_DYNAMIC_FRICTION = 2
HIGH_RESTITUTION = 0

APPLY_TO_OBJECT_COLLIDERS = True
BIND_PHYSICS_MATERIAL_VIA_KIT_COMMAND = True
DEBUG_PRINT_PHYSICS_BINDINGS = True

GRASPS_PER_PART = 6000
MAX_PARTS = 10
DISPLACEMENT_THRESHOLD = 0.03
DISTANCE_THRESHOLD = 0.005
APPROACH_DIST = 0.3  
RETRIEVAL_DIST = 0.30  
MOVE_STEPS = 200
MAX_Z_AXIS_DEVIATION = 30.0 
MIN_HORIZONTAL_PENETRATION_PERCENT = 10
MIN_Z_PENETRATION_PERCENT = 10 
HOLD_TEST_STEPS = 300              
HOLD_FALL_Z_DROP_THRESH = 0.05     
CONTACTSLOPCOEFF = 2.0

#Cloner
NUM_COPIES = 400
ENV_BASE_PATH = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"
CLONE_SPACING = 3.0
GROUND_Z = -10.0

#Pertabation
RETRIEVAL_PERTURBATION_AMP_DEG = 0.0
RETRIEVAL_PERTURBATION_FREQ    = 0.5
RETRIEVAL_PERTURBATION_DELAY_STEPS = 30

#Retrieval Forces
RETRIEVAL_FORCE_TEST_MODE = "all"
# options: "none", "all", "pos_x", "neg_x", "pos_y", "neg_y", "neg_z"
RETRIEVAL_FORCE_UNITS = "newton"
# "body_weight": force magnitude = RETRIEVAL_FORCE_MAG * m * g
# "newton": force magnitude = RETRIEVAL_FORCE_MAG in Newtons
RETRIEVAL_FORCE_MAG = 2.0
GRAVITY_ACCEL = 9.81
RETRIEVAL_FORCE_ALL_MODES = ("neg_z", "pos_x", "neg_x", "pos_y", "neg_y")

#Manual penetration thresholds for specific object categories and parts
PENETRATION_THRESHOLDS = {
    ("Mug", "handle"): (70, 40),  
    ("Mug", "body"): (0, 50),
    ("Bottle", "body"): (60, 6),
    ("Bottle", "lid"): (70, 50),
    ("Bottle", "neck"): (70, 10),
    ("Bottle", "mouth"): (70, 10),
    ("Knife", "blade_side"): (70, 10),
    ("Knife", "handle_side"): (70, 10),
    ("Scissors", "blade"): (50, 70),
    ("Scissors", "handle"): (40, 70),
    ("Hat", "bill"): (50, 60),
    ("Hat", "panel"): (0, 50),
    ("Hat", "crown"): (50, 10),
    ("Hat", "brim"): (20, 70),
    ("Vase", "body"): (60, 6),
    ("Headphone", "head_band"): (50, 10),
    ("Headphone", "earcup_unit"): (20, 50),
    ("Headphone", "earbud_connector_wire"): (20, 60),
    ("Headphone", "connector_wire"): (20, 60),
    ("Bowl", "container"): (20, 50)
}

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

# Path structure: wrapper (non-instanceable) contains ref (will become instanceable)
OBJECT_WRAPPER_PATH = obj_wrap(0)
OBJECT_REF_PATH = obj_ref(0)
GRIPPER_WRAPPER_PATH = grip_wrap(0)
GRIPPER_REF_PATH = grip_ref(0)

# Paths inside the references
GRIPPER_BASE_PATH = grip_base(0)

PROBES = [
    f"{GRIPPER_REF_PATH}/panda_leftfinger/probeA",
    f"{GRIPPER_REF_PATH}/panda_leftfinger/ProbeE",
    f"{GRIPPER_REF_PATH}/panda_leftfinger/ProbeF",
    f"{GRIPPER_REF_PATH}/panda_leftfinger/ProbeI",
    f"{GRIPPER_REF_PATH}/panda_leftfinger/ProbeJ",
    f"{GRIPPER_REF_PATH}/panda_rightfinger/probeB",
    f"{GRIPPER_REF_PATH}/panda_rightfinger/ProbeC",
    f"{GRIPPER_REF_PATH}/panda_rightfinger/ProbeD",
    f"{GRIPPER_REF_PATH}/panda_rightfinger/ProbeG",
    f"{GRIPPER_REF_PATH}/panda_rightfinger/ProbeH",
]

# =======================
# PArt Utilities
# =======================
def compute_transform_from_poses(initial_pos: Gf.Vec3d, initial_quat: Gf.Quatd,
                                  final_pos: Gf.Vec3d, final_quat: Gf.Quatd) -> tuple:
    """
    Compute relative transform from initial to final pose.
    Returns: (delta_pos, delta_rotation)
    """
    delta_pos = final_pos - initial_pos
    
    # Convert quaternions to rotation matrices
    initial_rot = Gf.Rotation(initial_quat)
    final_rot = Gf.Rotation(final_quat)
    
    # Compute relative rotation: final = delta * initial
    # So delta = final * initial^-1
    initial_rot_inv = initial_rot.GetInverse()
    delta_rot = final_rot * initial_rot_inv
    
    return delta_pos, delta_rot

def apply_inverse_transform_to_point(point: np.ndarray, 
                                      delta_pos: Gf.Vec3d, 
                                      delta_rot: Gf.Rotation) -> np.ndarray:
    """
    Apply inverse transform to bring point from final frame back to initial frame.
    Used to transform probe positions back to object's original frame.
    """
    # Convert numpy point to Gf.Vec3d
    pt = Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
    
    # First, subtract translation
    pt_translated = pt - delta_pos
    
    # Then, apply inverse rotation
    delta_rot_inv = delta_rot.GetInverse()
    pt_rotated = delta_rot_inv.TransformDir(pt_translated)
    
    return np.array([float(pt_rotated[0]), float(pt_rotated[1]), float(pt_rotated[2])])

def apply_transform_to_bbox(bbox_min: np.ndarray, bbox_max: np.ndarray,
                             delta_pos: Gf.Vec3d, delta_rot: Gf.Rotation) -> tuple:
    """
    Apply transform to bounding box (for penetration check).
    Transforms bbox from initial frame to final frame.
    """
    # Transform all 8 corners - CONVERT numpy.float32 to Python float
    corners = [
        Gf.Vec3d(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        Gf.Vec3d(float(bbox_max[0]), float(bbox_min[1]), float(bbox_min[2])),
        Gf.Vec3d(float(bbox_min[0]), float(bbox_max[1]), float(bbox_min[2])),
        Gf.Vec3d(float(bbox_min[0]), float(bbox_min[1]), float(bbox_max[2])),
        Gf.Vec3d(float(bbox_max[0]), float(bbox_max[1]), float(bbox_min[2])),
        Gf.Vec3d(float(bbox_max[0]), float(bbox_min[1]), float(bbox_max[2])),
        Gf.Vec3d(float(bbox_min[0]), float(bbox_max[1]), float(bbox_max[2])),
        Gf.Vec3d(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
    ]
    
    transformed_corners = []
    for corner in corners:
        # Apply rotation then translation
        rotated = delta_rot.TransformDir(corner)
        transformed = rotated + delta_pos
        transformed_corners.append([float(transformed[0]), float(transformed[1]), float(transformed[2])])
    
    transformed_np = np.array(transformed_corners, dtype=np.float64)  # Use float64 for consistency
    new_min = transformed_np.min(axis=0)
    new_max = transformed_np.max(axis=0)
    
    return new_min, new_max

def precompute_mesh_samples(stage, object_ref_path: str, max_points_per_mesh: int = 8000) -> dict:
    """Sample mesh vertices in LOCAL (template) coordinates before instancing"""
    mesh_samples = {}
    
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    if not ref_prim.IsValid():
        return mesh_samples
    
    for prim in Usd.PrimRange(ref_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        if len(points) == 0:
            continue
        
        # Downsample if too many points
        if len(points) > max_points_per_mesh:
            step = max(1, len(points) // max_points_per_mesh)
            points = points[::step]
        
        # Store in LOCAL coordinates
        local_points = np.array([[float(p[0]), float(p[1]), float(p[2])] 
                                  for p in points], dtype=np.float32)
        
        # Relative path for matching across clones
        rel_path = str(prim.GetPath()).replace(object_ref_path, "")
        mesh_samples[rel_path] = local_points
    
    return mesh_samples

def get_parts_list(stage, object_ref_path: str) -> list:
    """
    Get list of (part_path, part_name) tuples.
    Structure: object_ref_path -> child (e.g., /World or /Scan) -> parts
    """
    parts = []
    ref_prim = stage.GetPrimAtPath(object_ref_path)
    
    if not ref_prim.IsValid():
        return parts
    
    # Get children of object_ref
    children = list(ref_prim.GetChildren())
    
    if len(children) == 0:
        return parts

    parts_container = None
    for child in children:
        # Look for a prim that has mesh-containing children
        grandchildren = list(child.GetChildren())
        if len(grandchildren) > 1:  # Multiple parts indicate this is the container
            parts_container = child
            break
    
    if not parts_container:
        # Fallback: if only one child, check if it has parts
        if len(children) == 1:
            parts_container = children[0]
    
    if not parts_container:
        return parts
    
    # Now get parts from the container
    for part_prim in parts_container.GetChildren():
        if not (part_prim.IsA(UsdGeom.Xform) or part_prim.IsA(UsdGeom.Boundable)):
            continue
        
        has_mesh = any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(part_prim))
        if not has_mesh:
            continue
        
        part_name = part_prim.GetName()
        
        # Skip parts with "new" or "original" in name
        if should_ignore_part(part_name):
            continue
        
        part_path = part_prim.GetPath().pathString
        parts.append((part_path, part_name))
    
    if len(parts) > MAX_PARTS:
        print(f"[INFO] Limiting parts from {len(parts)} to {MAX_PARTS}")
        parts = parts[:MAX_PARTS]
    
    return parts

def compute_probe_offsets_in_gripper_frame(stage, gripper_ref_path: str) -> np.ndarray:
    """Compute probe positions relative to gripper base in LOCAL gripper coordinates"""
    gripper_base_path = f"{gripper_ref_path}/panda_hand"
    gripper_base = stage.GetPrimAtPath(gripper_base_path)
    
    if not gripper_base.IsValid():
        return np.empty((0, 3), dtype=np.float32)
    
    probe_offsets = []
    for probe_path in PROBES:
        probe_prim = stage.GetPrimAtPath(probe_path)
        if not probe_prim.IsValid():
            continue
        
        # Get transforms
        base_xform = UsdGeom.Xformable(gripper_base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        probe_xform = UsdGeom.Xformable(probe_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Probe position in gripper-local frame
        probe_world = probe_xform.Transform(Gf.Vec3d(0, 0, 0))
        base_inv = base_xform.GetInverse()
        probe_local = base_inv.Transform(probe_world)
        
        probe_offsets.append([float(probe_local[0]), float(probe_local[1]), float(probe_local[2])])
    
    return np.array(probe_offsets, dtype=np.float32)

def transform_probes_by_grasp_pose(probe_offsets: np.ndarray, 
                                    grasp_pos: Gf.Vec3d, 
                                    grasp_quat: Gf.Quatd) -> np.ndarray:
    """Transform probe offsets from gripper frame to object-local frame"""
    rot = Gf.Rotation(grasp_quat)
    probe_positions = np.empty_like(probe_offsets)
    
    for i, offset in enumerate(probe_offsets):
        offset_vec = Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2]))
        rotated = rot.TransformDir(offset_vec)
        world_pos = grasp_pos + rotated
        probe_positions[i] = [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])]
    
    return probe_positions


def batch_match_probes_to_meshes_local(probe_positions: np.ndarray,
                                       mesh_samples: dict) -> list:
    """Match probes to meshes - all in same local coordinate frame"""
    results = []
    
    for probe_pos in probe_positions:
        best_mesh = None
        best_dist = np.inf
        
        for mesh_rel_path, points in mesh_samples.items():
            if points.shape[0] == 0:
                continue
            
            diffs = points - probe_pos[None, :]
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            min_dist = float(np.min(distances))
            
            if min_dist < best_dist:
                best_dist = min_dist
                best_mesh = mesh_rel_path
        
        results.append((best_mesh, best_dist))
    
    return results

def extract_part_from_mesh_path(mesh_rel_path: str, parts_list: list) -> str:
    """
    Extract part name from relative mesh path.
    Match against known parts to find which part this mesh belongs to.
    """
    if not mesh_rel_path:
        return "body"
    
    # Try to match mesh path against known parts
    for part_path, part_name in parts_list:
        # Check if mesh path contains the part name
        if part_name in mesh_rel_path:
            return part_name
    
    # Fallback: try to extract from path structure
    parts = mesh_rel_path.strip("/").split("/")
    # Could be /World/Scan/handle/mesh or /Scan/handle/mesh or /category/handle/mesh
    # Look for the deepest non-mesh name
    for i in range(len(parts) - 1, -1, -1):
        part_candidate = parts[i]
        if not part_candidate.startswith("mesh") and not part_candidate.startswith("Mesh"):
            # Check if this matches any known part
            for _, part_name in parts_list:
                if part_candidate == part_name or part_candidate.lower() == part_name.lower():
                    return part_name
    
    return "body"

# =======================
# Helper Functions
# =======================

@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def should_ignore_part(part: str) -> bool:
    if part is None:
        return True
    part_lower = part.lower()
    return "new" in part_lower or "original" in part_lower

def load_functional_pairs(path: Path | None):
    if path is None or not path.exists():
        return None
    obj = load_json(path)
    pairs = obj.get("functional_pairs", [])
    out = set()
    for pair in pairs:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError("Each functional pair must be a 2-element list [type, part]")
        out.add((pair[0], pair[1]))
    return out

def get_penetration_thresholds(obj_cat: str, part_name: str) -> tuple[float, float]:
    part_lower = part_name.lower()
    obj_cat_lower = obj_cat.lower()
    
    key = (obj_cat, part_name)
    if key in PENETRATION_THRESHOLDS:
        return PENETRATION_THRESHOLDS[key]
    
    for (cat, part), thresholds in PENETRATION_THRESHOLDS.items():
        if cat.lower() == obj_cat_lower and part.lower() == part_lower:
            return thresholds
    
    return (MIN_HORIZONTAL_PENETRATION_PERCENT, MIN_Z_PENETRATION_PERCENT)

def extract_pose7(pose_obj):
    pos = pose_obj.get("position", [None, None, None])
    ori = pose_obj.get("orientation", {})
    qw = ori.get("w", None)
    xyz = ori.get("xyz", [None, None, None])
    if any(v is None for v in pos) or qw is None or any(v is None for v in xyz):
        raise ValueError("Pose is missing position/orientation fields")
    return [pos[0], pos[1], pos[2], qw, xyz[0], xyz[1], xyz[2]]

def create_yaml_config(yaml_path: Path, object_wrapper_path: str, gripper_wrapper_path: str):
    gripper_ref_path = f"{gripper_wrapper_path}/ref"
    
    config = {
        "object_path": object_wrapper_path,
        "gripper_path": gripper_wrapper_path,
        "num_orientations": 3,
        "joint_pregrasp_states": {
            f"{gripper_ref_path}/panda_hand/panda_finger_joint1": 0.039737965911626816,
            f"{gripper_ref_path}/panda_hand/panda_finger_joint2": 0.03973797708749771
        },
        "sampler_config": {
            "num_candidates": 500,
            "num_orientations": 3,
            "grasp_align_axis": [0, 1, 0],
            "orientation_sample_axis": [0, 1, 0],
            "gripper_approach_direction": [0, 0, 1],
            "gripper_maximum_aperture": 0.08,
            "gripper_standoff_fingertips": 0.17,
            "lateral_sigma": 0.02,
            "random_seed": 42,
            "sampler_type": "antipodal",
            "verbose": False
        },
        "grasp_phases": [
            {
                "name": "Open",
                "simulation_steps": 32,
                "simulation_step_dt": 0.016666666666666666,
                "joint_drive_targets": {
                    f"{gripper_ref_path}/panda_hand/panda_finger_joint1": 0.04
                }
            },
            {
                "name": "Close",
                "simulation_steps": 32,
                "simulation_step_dt": 0.016666666666666666,
                "joint_drive_targets": {
                    f"{gripper_ref_path}/panda_hand/panda_finger_joint1": 0.0
                }
            }
        ]
    }
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path

# =======================
# USD Processing Functions
# =======================
def get_bbox_center(stage, prim_path: str):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.GetRange()
    center = bbox_range.GetMidpoint()
    return Gf.Vec3d(center[0], center[1], center[2])

def center_object_at_origin(stage, obj_root_path: str):
    obj_prim = stage.GetPrimAtPath(obj_root_path)
    if not obj_prim.IsValid():
        print(f"[ERROR] Object prim not found at {obj_root_path}")
        return None
    
    center = get_bbox_center(stage, obj_root_path)
    if center is None:
        print(f"[ERROR] Could not compute bbox for {obj_root_path}")
        return obj_root_path
    
    tolerance = 0.001
    if abs(center[0]) < tolerance and abs(center[1]) < tolerance and abs(center[2]) < tolerance:
        print(f"[INFO] Object already centered at origin")
        return obj_root_path
    
    print(f"[INFO] Object center at ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), recentering...")
    
    original_name = obj_prim.GetName()
    parent_name = "Object"  
    
    parent_path = f"/{parent_name}"

    if stage.GetPrimAtPath(parent_path).IsValid() and parent_path != obj_root_path:
        parent_name = f"{original_name}_Centered"
        parent_path = f"/{parent_name}"
    
    parent_xform = UsdGeom.Xform.Define(stage, parent_path)
    parent_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    
    print(f"[INFO] Created parent xform at {parent_path} with translation (0,0,0)")
    
    child_name = original_name if original_name != parent_name else f"{original_name}_geo"
    child_path = f"{parent_path}/{child_name}"
    
    root_layer = stage.GetRootLayer()
    Sdf.CopySpec(root_layer, obj_root_path, root_layer, child_path)
    
    print(f"[INFO] Copied {obj_root_path} to {child_path}")
    
    child_prim = stage.GetPrimAtPath(child_path)
    child_xform = UsdGeom.Xformable(child_prim)
    
    existing_ops = child_xform.GetOrderedXformOps()
    translate_op = None
    
    for op in existing_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    
    centering_offset = Gf.Vec3d(-center[0], -center[1], -center[2])
    
    if translate_op:
        existing_translation = translate_op.Get()
        if existing_translation is None:
            existing_translation = Gf.Vec3d(0, 0, 0)
        new_translation = existing_translation + centering_offset
        translate_op.Set(new_translation)
        print(f"[INFO] Updated child translate to {new_translation}")
    else:
        translate_op = child_xform.AddTranslateOp()
        translate_op.Set(centering_offset)
        print(f"[INFO] Created child translate: {centering_offset}")
    
    update_material_references(stage, obj_root_path, parent_path, child_path)
    
    if obj_root_path != parent_path:
        stage.RemovePrim(obj_root_path)
        print(f"[INFO] Removed original prim at {obj_root_path}")
    
    print(f"[INFO] Object recentered - parent at {parent_path} with translation (0,0,0)")
    return parent_path

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

def update_material_references(stage, old_root_path: str, new_parent_path: str, new_child_path: str):
    print(f"[INFO] Updating material references...")
    print(f"[DEBUG] old_root_path: {old_root_path}")
    print(f"[DEBUG] new_parent_path: {new_parent_path}")
    print(f"[DEBUG] new_child_path: {new_child_path}")
    
    root_layer = stage.GetRootLayer()
    
    looks_paths = [
        f"{old_root_path}/Looks",
        f"{old_root_path}/Materials", 
        "/Looks",
        "/Materials"
    ]
    
    for looks_path in looks_paths:
        looks_prim = stage.GetPrimAtPath(looks_path)
        if not looks_prim.IsValid():
            continue
        
        print(f"[INFO] Found materials at: {looks_path} (already copied to {new_child_path})")
    
    new_child_prim = stage.GetPrimAtPath(new_child_path)
    if not new_child_prim.IsValid():
        print(f"[WARN] New child prim not found at {new_child_path}")
        return
    
    for prim in Usd.PrimRange(new_child_prim):
        if prim.HasRelationship("material:binding"):
            binding_rel = prim.GetRelationship("material:binding")
            targets = binding_rel.GetTargets()
            
            updated_targets = []
            for target in targets:
                target_str = str(target)
                
                if target_str.startswith(old_root_path):
                    new_target_str = target_str.replace(old_root_path, new_child_path, 1)
                    updated_targets.append(Sdf.Path(new_target_str))
                    print(f"[INFO] Updated binding: {target_str} -> {new_target_str}")
                else:
                    updated_targets.append(target)
            
            if updated_targets:
                binding_rel.SetTargets(updated_targets)
    
    print(f"[INFO] Material references updated")

def remove_rigid_body_from_usd(stage, root_path: str):
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[WARN] Cannot remove RigidBody - invalid prim: {root_path}")
        return False
    
    total_removed = 0
    
    for prim in Usd.PrimRange(root_prim):
        removed_from_prim = []
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed_from_prim.append("RigidBodyAPI")
        
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            removed_from_prim.append("PhysxRigidBodyAPI")
        
        if removed_from_prim:
            print(f"[INFO]   Removed {', '.join(removed_from_prim)} from {prim.GetPath()}")
            total_removed += 1
    
    if total_removed > 0:
        print(f"[INFO] Removed RigidBody APIs from {total_removed} prim(s) under {root_path}")
        return True
    else:
        print(f"[INFO] No RigidBodyAPI found under {root_path}")
        return False

def _prim_subtree_has_mesh(root_prim: Usd.Prim) -> bool:
    if not root_prim or not root_prim.IsValid():
        return False
    return any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(root_prim))

def _resolve_geom_root_under_path(stage, root_path: str) -> str:
    """
    Prefer /Scan when present, otherwise fall back to the only child directly
    under root_path when assets are authored as /ref/<single_child>/...
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return root_path

    scan_path = f"{root_path}/Scan"
    scan_prim = stage.GetPrimAtPath(scan_path)
    if scan_prim.IsValid() and _prim_subtree_has_mesh(scan_prim):
        return scan_path

    children = [child for child in root_prim.GetChildren() if child.IsValid()]
    if len(children) == 1 and _prim_subtree_has_mesh(children[0]):
        return children[0].GetPath().pathString

    return root_path

def check_object_needs_physics(usd_path: Path) -> bool:
    try:
        temp_stage = Usd.Stage.Open(str(usd_path))
        if not temp_stage:
            return True
        
        geom_root_path = _resolve_geom_root_under_path(temp_stage, "/World")
        scan_prim = temp_stage.GetPrimAtPath(geom_root_path)
        if not scan_prim or not scan_prim.IsValid():
            return True
        
        # Check if any mesh has collision API
        found_mesh_with_collision = False
        for p in Usd.PrimRange(scan_prim):
            if p.IsA(UsdGeom.Mesh):
                if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    found_mesh_with_collision = True
                    break
        
        return not found_mesh_with_collision
        
    except Exception as e:
        print(f"[WARN] Could not check {usd_path}: {e}")
        return True

def _ensure_convex_decomposition_on_meshes(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    mesh_count = 0
    
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        
        mesh_count += 1
        
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        
        mapi = UsdPhysics.MeshCollisionAPI(p)
        if not mapi:
            mapi = UsdPhysics.MeshCollisionAPI.Apply(p)
        
        approx = mapi.CreateApproximationAttr()
        approx.Set(UsdPhysics.Tokens.convexDecomposition)
    
    print(f"[INFO] Applied convex decomposition to {mesh_count} meshes")

def setup_physics_and_collision(stage, obj_root: str = "/World"):
    print(f"[INFO] Setting up physics and collision for {obj_root}")
    
    scan_path = _resolve_geom_root_under_path(stage, obj_root)
    scan_prim = stage.GetPrimAtPath(scan_path)
    if not scan_prim.IsValid():
        print(f"[ERROR] Invalid root prim: {obj_root}")
        return
    
    _ensure_convex_decomposition_on_meshes(stage, scan_path)
    
    print(f"[INFO] Physics setup complete - collision meshes configured on {scan_path}")

def set_disable_gravity(stage, prim_path: str, disable: bool):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found for gravity toggle: {prim_path}")
    
    if prim.IsInstanceable():
        raise RuntimeError(
            f"Cannot modify gravity on instanceable prim {prim_path}! "
            f"Use the non-instanceable wrapper instead."
        )
    
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateDisableGravityAttr().Set(bool(disable))


# =======================
# RETRIEVAL FORCE HELPERS
# =======================
def _to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)

def _get_retrieval_force_direction(mode):
    if mode == "pos_x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if mode == "neg_x":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    if mode == "pos_y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if mode == "neg_y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if mode == "neg_z":
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def _resolve_retrieval_force_mode(mode, frame_idx=None, total_frames=None):
    if mode != "all":
        return mode
    if total_frames is None or total_frames <= 0 or frame_idx is None:
        raise ValueError("'all' retrieval force mode requires frame_idx and total_frames")
    segment_idx = min(
        int(frame_idx * len(RETRIEVAL_FORCE_ALL_MODES) / float(total_frames)),
        len(RETRIEVAL_FORCE_ALL_MODES) - 1,
    )
    return RETRIEVAL_FORCE_ALL_MODES[segment_idx]

def build_simple_retrieval_force_array(
    active,
    num_copies,
    mode,
    mag,
    units="newton",
    masses=None,
    frame_idx=None,
    total_frames=None,
):
    forces = np.zeros((num_copies, 3), dtype=np.float32)
    resolved_mode = _resolve_retrieval_force_mode(mode, frame_idx=frame_idx, total_frames=total_frames)
    direction = _get_retrieval_force_direction(resolved_mode)
    if not np.any(direction):
        return forces

    if units == "body_weight":
        if masses is None:
            raise ValueError("masses are required when RETRIEVAL_FORCE_UNITS == 'body_weight'")
        magnitudes = _to_numpy(masses).reshape(num_copies) * float(mag) * GRAVITY_ACCEL
    else:
        magnitudes = np.full((num_copies,), float(mag), dtype=np.float32)

    for k in range(min(len(active), num_copies)):
        if active[k]:
            forces[k] = direction * magnitudes[k]

    return forces

def create_object_force_tensor_view(num_copies: int):
    backend = SimulationManager.get_backend()
    sim_view = omni.physics.tensors.create_simulation_view(
        backend,
        stage_id=get_current_stage_id(),
    )
    sim_view.set_subspace_roots("/")

    rigid_body_view = sim_view.create_rigid_body_view([obj_wrap(i) for i in range(num_copies)])
    if rigid_body_view.count != num_copies:
        raise RuntimeError(
            f"object_force_view count mismatch: expected {num_copies}, got {rigid_body_view.count}"
        )

    return {
        "sim_view": sim_view,
        "rigid_body_view": rigid_body_view,
        "indices": np.arange(num_copies, dtype=np.uint32),
        "masses": _to_numpy(rigid_body_view.get_masses()).reshape(num_copies),
        "num_copies": num_copies,
    }

def apply_retrieval_forces(force_view_state, forces):
    try:
        force_view_state["rigid_body_view"].apply_forces(
            forces,
            force_view_state["indices"],
            True,
        )
        return force_view_state
    except Exception as exc:
        msg = str(exc)
        if "Failed to apply forces" not in msg and "invalidated" not in msg:
            raise

        rebound_state = create_object_force_tensor_view(force_view_state["num_copies"])
        rebound_state["rigid_body_view"].apply_forces(
            forces,
            rebound_state["indices"],
            True,
        )
        return rebound_state


def _get_or_create_xform(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return prim
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def _get_or_create_attr_local(prim, name: str, sdf_type):
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, sdf_type)
    return attr

def create_or_update_physx_material(stage,
                                   material_path: str,
                                   static_friction: float,
                                   dynamic_friction: float,
                                   restitution: float = 0.0):
    parent = "/".join(material_path.split("/")[:-1])
    if parent and parent != "":
        _get_or_create_xform(stage, parent)

    mat = UsdShade.Material.Define(stage, material_path)
    mat_prim = mat.GetPrim()

    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

    try:
        physx_mat.CreateStaticFrictionAttr().Set(float(static_friction))
    except Exception:
        _get_or_create_attr_local(mat_prim, "physxMaterial:staticFriction", Sdf.ValueTypeNames.Float).Set(float(static_friction))

    try:
        physx_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    except Exception:
        _get_or_create_attr_local(mat_prim, "physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(float(dynamic_friction))

    try:
        physx_mat.CreateRestitutionAttr().Set(float(restitution))
    except Exception:
        _get_or_create_attr_local(mat_prim, "physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(float(restitution))

    for attr_name, token_val in (
        ("physxMaterial:frictionCombineMode", "multiply"),
        ("physxMaterial:restitutionCombineMode", "multiply"),
    ):
        a = mat_prim.GetAttribute(attr_name)
        if a and a.IsValid():
            a.Set(token_val)

    return mat

def bind_physics_material(target_prim: Usd.Prim, material: UsdShade.Material):
    if not target_prim or not target_prim.IsValid():
        return

    api = UsdShade.MaterialBindingAPI(target_prim)

    try:
        api.Bind(material, materialPurpose="physics")
        return
    except TypeError:
        pass

    try:
        api.Bind(material, purpose="physics")
        return
    except TypeError:
        pass

    api.Bind(material)


def bind_physics_material_via_kit_command(target_prim: Usd.Prim, material: UsdShade.Material) -> bool:
    if not target_prim or not target_prim.IsValid():
        return False

    prim_path = target_prim.GetPath().pathString
    mat_path = material.GetPath().pathString

    attempts = [
        dict(prim_path=prim_path, material_path=mat_path, strength=None, material_purpose="physics"),
        dict(prim_path=prim_path, material_path=mat_path, strength=None, materialPurpose="physics"),
        dict(prim_path=prim_path, material_path=mat_path, strength=None, materialPurpose="physics"),
        dict(prim_path=prim_path, material_path=mat_path, strength=None),
    ]

    for kwargs in attempts:
        try:
            omni.kit.commands.execute("BindMaterialCommand", **kwargs)
            return True
        except Exception:
            continue

    return False


def debug_print_physics_material_bindings(stage, root_path: str, max_print: int = 20):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[DEBUG] debug_print_physics_material_bindings: invalid root {root_path}")
        return

    rel_names = ["material:binding", "material:binding:physics"]
    found = 0

    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue

        for rn in rel_names:
            rel = p.GetRelationship(rn)
            if rel and rel.IsValid():
                targets = rel.GetTargets()
                if targets:
                    if found < max_print:
                        print(f"[DEBUG] {p.GetPath()} has {rn} -> {[str(t) for t in targets]}")
                    found += 1

    print(f"[DEBUG] Total mesh prims with any material binding rel under {root_path}: {found}")


def apply_physics_material_to_collision_meshes(stage, root_path: str, material: UsdShade.Material) -> int:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[WARN] apply_physics_material_to_collision_meshes: invalid root {root_path}")
        return 0

    count = 0
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue

        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)

        if p.HasAPI(UsdPhysics.CollisionAPI):
            if BIND_PHYSICS_MATERIAL_VIA_KIT_COMMAND:
                ok = bind_physics_material_via_kit_command(p, material)
                if not ok:
                    bind_physics_material(p, material)
            else:
                bind_physics_material(p, material)
            count += 1

    return count

async def ensure_timeline_playing():
    """Ensure timeline is playing - force restart if stopped"""
    if not timeline.is_playing():
        print(f"[DEBUG] Timeline stopped, restarting...")
        timeline.play()
        await omni.kit.app.get_app().next_update_async()

def flatten_and_save_usd(stage, output_path: Path, object_path: str):

    print(f"[INFO] Flattening and saving {object_path} to {output_path}")
    
    obj_prim = stage.GetPrimAtPath(object_path)
    if not obj_prim.IsValid():
        print(f"[ERROR] Object prim not found at {object_path}")
        return False
    
    prim_name = obj_prim.GetName()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    flattened_layer = stage.Flatten()
    
    export_stage = Usd.Stage.CreateNew(str(output_path))
    export_root_layer = export_stage.GetRootLayer()
    
    Sdf.CopySpec(
        flattened_layer,
        object_path,                 
        export_root_layer,
        f"/{prim_name}"              
    )
    
    export_stage.SetDefaultPrim(export_stage.GetPrimAtPath(f"/{prim_name}"))
    
    src_up = UsdGeom.GetStageUpAxis(stage)
    UsdGeom.SetStageUpAxis(export_stage, src_up)

    src_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    UsdGeom.SetStageMetersPerUnit(export_stage, src_mpu)
    
    export_root_layer.Save()
    
    print(f"[INFO] Saved flattened USD with top prim: /{prim_name}")
    print(f"[INFO] UpAxis = {src_up}, metersPerUnit = {src_mpu}")
    return True

def get_geom_root_path(stage, object_wrapper_path: str) -> str:
    """Get path to geometry - checks if reference is instanceable"""
    ref_path = f"{object_wrapper_path}/ref"
    ref_prim = stage.GetPrimAtPath(ref_path)
    
    # If instanceable, return wrapper (can't access inside)
    if ref_prim.IsValid() and ref_prim.IsInstanceable():
        return object_wrapper_path
    
    return _resolve_geom_root_under_path(stage, ref_path)

async def preprocess_usd_file(input_usd: Path, output_usd: Path) -> tuple[bool, str]:

    print(f"\n{'='*80}")
    print(f"Preprocessing: {input_usd.name}")
    print(f"{'='*80}")
    
    ctx = omni.usd.get_context()
    
    print(f"[INFO] Opening {input_usd}")
    await ctx.open_stage_async(str(input_usd))
    await omni.kit.app.get_app().next_update_async()
    
    stage = ctx.get_stage()
    if not stage:
        print(f"[ERROR] Failed to open stage")
        return False, ""
    
    world_path = "/World"
    world_prim = stage.GetPrimAtPath(world_path)
    
    if not world_prim.IsValid():
        print(f"[ERROR] No /World prim found in Object.usd")
        return False, ""
    
    print(f"[INFO] Found /World in Object.usd")
    
    if not BYPASS_RECENTER_AND_FLATTEN:
        new_world_path = center_object_at_origin(stage, world_path)
        if new_world_path is None:
            return False, ""
        
        world_path = new_world_path
        world_prim = stage.GetPrimAtPath(world_path)
        
        await omni.kit.app.get_app().next_update_async()
    else:
        print(f"[INFO] BYPASS_RECENTER_AND_FLATTEN enabled - skipping centering")
    
    setup_physics_and_collision(stage, world_path)
    
    await omni.kit.app.get_app().next_update_async()

    if not BYPASS_RECENTER_AND_FLATTEN:
        flatten_and_save_usd(stage, output_usd, world_path)
    else:
        print(f"[INFO] BYPASS_RECENTER_AND_FLATTEN enabled - skipping flattening, saving directly")
        stage.GetRootLayer().Export(str(output_usd))
    
    print(f"Preprocessing complete: {output_usd}")
    print(f"Top prim: /World")
    return True, "World"


def apply_object_physx_overrides(stage, obj_wrapper_path: str):
    prim = stage.GetPrimAtPath(obj_wrapper_path)
    if not prim.IsValid():
        raise RuntimeError(f"[apply_object_physx_overrides] Invalid prim: {obj_wrapper_path}")

    if prim.IsInstanceable():
        raise RuntimeError(
            f"{obj_wrapper_path} is instanceable! Physics APIs must be on non-instanceable wrapper."
        )

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(True)

    physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)

    physx_rb.CreateDisableGravityAttr().Set(True)

    try:
        physx_rb.CreateContactSlopCoefficientAttr().Set(float(CONTACTSLOPCOEFF))
    except Exception:
        _get_or_create_attr_local(
            prim,
            "physxRigidBody:contactSlopCoefficient",
            Sdf.ValueTypeNames.Float
        ).Set(float(CONTACTSLOPCOEFF))

    print(f"[INFO] Applied runtime PhysX overrides on wrapper {obj_wrapper_path}: "
          f"disableGravity=True, contactSlopCoefficient={CONTACTSLOPCOEFF}")
    
# =======================
# Main Pipeline Functions
# =======================
async def run_pipeline():
    print(f"\n{'#'*80}")
    print(f"# Combined Grasp Generation + Classification Pipeline")
    print(f"{'#'*80}")
    print(f"Input directory: {INPUT_ROOT}")
    print(f"Output directory: {TARGET_OUTPUT_ROOT}")
    print(f"BYPASS_RECENTER_AND_FLATTEN: {BYPASS_RECENTER_AND_FLATTEN}")
    print(f"CLEAN_EXISTING_RIGID_BODY: {CLEAN_EXISTING_RIGID_BODY}")
    print(f"{'#'*80}\n")
    
    TARGET_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    objects_to_process = []
    
    for category_dir in INPUT_ROOT.iterdir():
        if not category_dir.is_dir():
            continue
        
        for obj_dir in category_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            
            object_usd = obj_dir / "Object.usd"
            if object_usd.exists():
                print(f"[INFO] Found preprocessed Object.usd in {obj_dir.relative_to(INPUT_ROOT)}")
                objects_to_process.append({
                    'preprocessed': True,
                    'object_usd': object_usd,
                    'obj_dir': obj_dir,
                    'obj_id': f"{obj_dir.name}"
                })
            else:
                raw_usd_files = list(obj_dir.glob("*.usd"))
                if raw_usd_files:
                    input_usd = raw_usd_files[0]
                    print(f"[INFO] Found raw USD {input_usd.name} in {obj_dir.relative_to(INPUT_ROOT)}")
                    objects_to_process.append({
                        'preprocessed': False,
                        'input_usd': input_usd,
                        'obj_dir': obj_dir,
                        'obj_id': f"{obj_dir.name}"
                    })
    
    if not objects_to_process:
        print(f"[WARN] No USD files found in {INPUT_ROOT}")
        return
    
    print(f"[INFO] Found {len(objects_to_process)} objects to process")
    print(f"  - {sum(1 for o in objects_to_process if o['preprocessed'])} already preprocessed")
    print(f"  - {sum(1 for o in objects_to_process if not o['preprocessed'])} need preprocessing\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, obj_info in enumerate(objects_to_process, 1):
        obj_id = obj_info['obj_id']
        obj_dir = obj_info['obj_dir']
        
        print(f"\n{'='*80}")
        print(f"Object {idx}/{len(objects_to_process)}: {obj_id}")
        print(f"{'='*80}")
        
        try:
            if obj_info['preprocessed']:
                print(f"[INFO] Using existing Object.usd (skipping preprocessing)")
                object_usd = obj_info['object_usd']
                
                if CLEAN_EXISTING_RIGID_BODY:
                    print(f"[INFO] Checking for baked RigidBodyAPI to remove...")
                    ctx = omni.usd.get_context()
                    await ctx.open_stage_async(str(object_usd))
                    stage = ctx.get_stage()
                    
                    if stage:
                        world_path = "/World"
                        removed = remove_rigid_body_from_usd(stage, world_path)
                        
                        if removed:
                            stage.GetRootLayer().Save()
                            print(f"[INFO] Saved cleaned Object.usd")
                
                elif not BYPASS_RECENTER_AND_FLATTEN:
                    needs_physics = check_object_needs_physics(object_usd)
                    if needs_physics:
                        print(f"[INFO] Object needs physics setup - will add during preprocessing check")
                        ctx = omni.usd.get_context()
                        await ctx.open_stage_async(str(object_usd))
                        stage = ctx.get_stage()
                        
                        if stage:
                            world_path = "/World"
                            setup_physics_and_collision(stage, world_path)
                            await omni.kit.app.get_app().next_update_async()
                            
                            if not BYPASS_RECENTER_AND_FLATTEN:
                                flatten_and_save_usd(stage, object_usd, world_path)
                                print(f"[INFO] Added physics, flattened and saved Object.usd")
                            else:
                                stage.GetRootLayer().Save()
                                print(f"[INFO] Added physics and saved Object.usd (no flattening)")
                    else:
                        print(f"[INFO] Object already has physics setup - skipping")
                else:
                    print(f"[INFO] BYPASS_RECENTER_AND_FLATTEN enabled - skipping physics check completely")
                
            else:
                print(f"[INFO] Preprocessing raw USD file")
                input_usd = obj_info['input_usd']
                object_usd = obj_dir / "Object.usd"
                
                success, top_prim_name = await preprocess_usd_file(input_usd, object_usd)
                if not success:
                    print(f"[ERROR] Preprocessing failed for {input_usd.name}")
                    fail_count += 1
                    continue
                
                try:
                    input_usd.unlink()
                    print(f"[INFO] Deleted original USD: {input_usd.name}")
                except Exception as e:
                    print(f"[WARN] Could not delete original USD {input_usd}: {e}")
            
            yaml_path = obj_dir / "grasp_config.yaml"
            create_yaml_config(yaml_path, OBJECT_WRAPPER_PATH, GRIPPER_WRAPPER_PATH)
            print(f"[INFO] Created YAML config at {yaml_path}")
            
            await process_one_object_async(object_usd, obj_id, obj_dir, yaml_path, OBJECT_WRAPPER_PATH)
            
            success_count += 1
            with open(LOG_FILE, "a") as f:
                f.write(f"{obj_id}\n")
                
        except Exception as e:
            print(f"\n!! Failed on {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print(f"\n{'#'*80}")
    print(f"# Pipeline Complete")
    print(f"{'#'*80}")
    print(f"Total: {len(objects_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"{'#'*80}\n")
  
async def process_one_object_async(obj_usd: Path, obj_id: str, output_dir: Path, yaml_path: Path, object_wrapper_path: str):
    print(f"\n{'='*80}")
    print(f"Processing grasps for: {obj_id}")
    print(f"Using wrapper structure:")
    print(f"  Wrapper: {object_wrapper_path} (non-instanceable, holds RigidBodyAPI)")
    print(f"  Reference: {OBJECT_REF_PATH} (will become instanceable after setup)")
    print(f"{'='*80}")
    
    ctx = omni.usd.get_context()
    if ctx.get_stage():
        print(f"[INFO] Closing previous stage...")
        await ctx.close_stage_async()
        await omni.kit.app.get_app().next_update_async()
        
        import gc
        gc.collect()
        
        # Let USD internals settle
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
        
    print(f"[INFO] Setting up object with wrapper structure...")
    object_wrapper_xform = UsdGeom.Xform.Define(stage, object_wrapper_path)
    print(f"[INFO]   Created wrapper at {object_wrapper_path}")
    
    object_ref_prim = add_reference_to_stage(str(obj_usd), OBJECT_REF_PATH)
    
    if not object_ref_prim:
        print(f"[ERROR] Failed to add object reference")
        return
    print(f"[INFO]   Added reference at {OBJECT_REF_PATH}")
    
    print(f"[INFO] Setting up gripper with wrapper structure...")
    
    gripper_wrapper_xform = UsdGeom.Xform.Define(stage, GRIPPER_WRAPPER_PATH)
    print(f"[INFO]   Created wrapper at {GRIPPER_WRAPPER_PATH}")
    
    gripper_ref_prim = add_reference_to_stage(str(GRIPPER_USD), GRIPPER_REF_PATH)
    if not gripper_ref_prim:
        print(f"[ERROR] Failed to add gripper reference")
        return
    print(f"[INFO]   Added reference at {GRIPPER_REF_PATH}")
    
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    
    print(f"[INFO] Removing any RigidBodyAPI from reference (before instanceable)...")
    ref_prim = stage.GetPrimAtPath(OBJECT_REF_PATH)
    if ref_prim.IsValid():
        cleaned = 0
        for prim in Usd.PrimRange(ref_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                cleaned += 1
                print(f"[DEBUG]   Removed UsdPhysics.RigidBodyAPI from {prim.GetPath()}")
            
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                cleaned += 1
                print(f"[DEBUG]   Removed PhysxSchema.PhysxRigidBodyAPI from {prim.GetPath()}")
        
        if cleaned > 0:
            print(f"[INFO] Removed {cleaned} RigidBodyAPI(s) from reference")
        else:
            print(f"[INFO] No RigidBodyAPI found in reference")

    await omni.kit.app.get_app().next_update_async()
    
    print(f"[INFO] Applying physics to wrapper...")
    apply_object_physx_overrides(stage, object_wrapper_path)
    await omni.kit.app.get_app().next_update_async()
    
    print(f"[INFO] Setting up physics scene...")
    physics_scene_path = setup_physics_scene(stage)
    await omni.kit.app.get_app().next_update_async()
    ps_prim = stage.GetPrimAtPath(physics_scene_path)
    if not ps_prim.IsValid():
        raise RuntimeError(f"Failed to create valid physics scene at {physics_scene_path}")
    print(f"[INFO] Physics scene validated at {physics_scene_path}")
    GroundPlane(prim_path = "/World/GroundPlane", z_position = GROUND_Z)
    await omni.kit.app.get_app().next_update_async()
    
    if APPLY_HIGH_FRICTION_MATERIAL and APPLY_TO_OBJECT_COLLIDERS:
        print(f"[INFO] Applying high friction material (before making instanceable)...")
        hf_mat = create_or_update_physx_material(
            stage,
            HIGH_FRICTION_MATERIAL_PATH,
            static_friction=HIGH_STATIC_FRICTION,
            dynamic_friction=HIGH_DYNAMIC_FRICTION,
            restitution=HIGH_RESTITUTION,
        )

        geom_root = get_geom_root_path(stage, object_wrapper_path)
        bound_obj = apply_physics_material_to_collision_meshes(stage, geom_root, hf_mat)
        print(f"[INFO] High-friction material bound: geom_root={geom_root}, meshes={bound_obj}")
        if DEBUG_PRINT_PHYSICS_BINDINGS:
            debug_print_physics_material_bindings(stage, geom_root, max_print=15)
        await omni.kit.app.get_app().next_update_async()
    
    meta_file = DATASET_PATH / obj_id / "meta.json"
    model_cat = "unknown"

    if meta_file.exists():
        meta = load_json(meta_file)
        model_cat = meta.get("model_cat", "unknown")

    if model_cat == "unknown":
        parent_dir = obj_usd.parent.parent.name
        if parent_dir and parent_dir != INPUT_ROOT.name:
            model_cat = parent_dir
            print(f"[INFO] Inferred category from directory: {model_cat}")
    
    print(f"[INFO] Model category: {model_cat}")
    
    #Part detection
    parts_list = []
    mesh_samples = {}
    has_parts = False
    if HAS_PARTS:
        print(f"\n[INFO] === PART DETECTION (before instancing) ===")
        parts_list = get_parts_list(stage, OBJECT_REF_PATH)
        has_parts = len(parts_list) > 1
        
        if has_parts:
            print(f"[INFO] Found {len(parts_list)} parts:")
            for part_path, part_name in parts_list:
                print(f"  - {part_name}")
            
            print(f"\n[INFO] Pre-computing geometry before instancing...")
            mesh_samples = precompute_mesh_samples(stage, OBJECT_REF_PATH, max_points_per_mesh=8000)

            total_mesh_points = sum(len(pts) for pts in mesh_samples.values())
            print(f"[INFO] Pre-computed:")
            print(f"  - {len(mesh_samples)} meshes ({total_mesh_points} total sample points)")
            
        else:
            print(f"[INFO] Only {len(parts_list)} part(s) found - processing as whole object")
    else:
        print(f"[INFO] USE_PARTS=False - processing as whole object only")
    
    bottom_center = get_bbox_bottom_center(stage, object_wrapper_path)
    if bottom_center:
        bottom_center_list = [float(bottom_center[0]), float(bottom_center[1]), float(bottom_center[2])]
        print(f"[INFO] Object bottom center: {bottom_center_list}")
    else:
        bottom_center_list = None
        print(f"[WARN] Could not compute bottom center")
    
    grasping_manager = GraspingManager()
    all_candidate_poses = []  # List of (loc, quat, target_part_name)

    if has_parts:
        print(f"\n[INFO] === GRASP GENERATION: Per-Part ===")
        
        for idx, (part_path, part_name) in enumerate(parts_list, 1):
            print(f"  Part {idx}/{len(parts_list)}: {part_name}")
            
            # Update YAML for this part
            load_and_modify_yaml_inplace(
                yaml_path, part_path, GRIPPER_WRAPPER_PATH, 
                GRASPS_PER_PART,  
                object_wrapper_path, part_name, None
            )
            
            if not grasping_manager.load_config(str(yaml_path)):
                print(f"    Failed to load config for {part_name}")
                continue
            
            if not grasping_manager.generate_grasp_poses():
                print(f"    No poses generated for {part_name}")
                continue
            
            part_poses = grasping_manager.get_grasp_poses(in_world_frame=True)
            part_poses = filter_bottom_up_grasps(part_poses)
            print(f"    Generated {len(part_poses)} poses for {part_name}")
            
            # Tag each pose with its target part
            for loc, quat in part_poses:
                all_candidate_poses.append((
                    Gf.Vec3d(float(loc[0]), float(loc[1]), float(loc[2])),
                    Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(*[float(v) for v in quat.GetImaginary()])),
                    part_name
                ))

    # Always add whole object grasps
    print(f"\n[INFO] === GRASP GENERATION: Whole Object ===")
    geom_root = get_geom_root_path(stage, object_wrapper_path)

    load_and_modify_yaml_inplace(
        yaml_path, geom_root, GRIPPER_WRAPPER_PATH, 
        GRASPS_PER_PART,
        object_wrapper_path, "WholeObject", None
    )

    if not grasping_manager.load_config(str(yaml_path)):
        print(f"    Failed to load config for WholeObject")
        return

    if not grasping_manager.generate_grasp_poses():
        print(f"    No poses generated for WholeObject")
        return

    whole_poses = grasping_manager.get_grasp_poses(in_world_frame=True)
    whole_poses = filter_bottom_up_grasps(whole_poses)
    print(f"    Generated {len(whole_poses)} poses for WholeObject")

    for loc, quat in whole_poses:
        all_candidate_poses.append((
            Gf.Vec3d(float(loc[0]), float(loc[1]), float(loc[2])),
            Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(*[float(v) for v in quat.GetImaginary()])),
            "WholeObject"
        ))

    print(f"\n[INFO] Total candidate poses: {len(all_candidate_poses)}")

    # Use all_candidate_poses instead of poses from here on
    poses = all_candidate_poses
    
    grasping_manager.store_initial_gripper_pose()
    obj_prim = grasping_manager.get_object_prim()
    if not obj_prim:
        print(f"    Could not get object prim")
        return
    print("[DEBUG] GraspingManager object prim path:", obj_prim.GetPath())
    
    close_phase = grasping_manager.get_grasp_phase_by_name("Close")
    close_targets = close_phase.joint_drive_targets if close_phase else {}
    if not close_targets:
        print("[WARN] 'Close' phase not found in YAML! Gripper won't close.")
    
    results = []
    grasps_rejected_approach_collision = 0 
    grasps_rejected_closure = 0
    grasps_rejected_retrieval = 0
    grasps_rejected_shallow = 0
    
    #Set instanceable after physics setup
    if has_parts and parts_list and len(parts_list) > 1:
        print(f"[INFO] Found {len(parts_list)} parts - making each one instanceable")
        
        for part_path, part_name in parts_list:
            part_prim = stage.GetPrimAtPath(part_path)
            if not part_prim.IsValid():
                print(f"[WARN] Part prim not found: {part_path}")
                continue
            
            # Make this part instanceable
            part_prim.SetInstanceable(True)
            print(f"[INFO]   ✓ Made part instanceable: {part_name} at {part_path}")
        
        # Keep the object reference itself NON-instanceable
        object_ref_prim_usd = stage.GetPrimAtPath(OBJECT_REF_PATH)
        if object_ref_prim_usd.IsValid():
            # Ensure it's NOT instanceable
            if object_ref_prim_usd.IsInstanceable():
                object_ref_prim_usd.SetInstanceable(False)
                print(f"[INFO]   Ensured {OBJECT_REF_PATH} is NON-instanceable (parts are instanceable)")
    else:
        print(f"[INFO] Single-part object - making geometry root instanceable")

        object_ref_prim_usd = stage.GetPrimAtPath(OBJECT_REF_PATH)
        if object_ref_prim_usd.IsValid():
            object_ref_prim_usd.SetInstanceable(True)
            print(f"[INFO]   {OBJECT_REF_PATH} is now instanceable")
        else:
            print(f"[ERROR] Could not make {OBJECT_REF_PATH} instanceable - prim not found")
        
        await omni.kit.app.get_app().next_update_async()
        
        print(f"[INFO] Verifying structure...")
        wrapper_prim = stage.GetPrimAtPath(object_wrapper_path)
        ref_prim = stage.GetPrimAtPath(OBJECT_REF_PATH)
        
        if wrapper_prim.IsInstanceable():
            print(f"[ERROR] Wrapper {object_wrapper_path} should NOT be instanceable!")
        else:
            print(f"[INFO]   ✓ Wrapper {object_wrapper_path} is non-instanceable")
        
        if not ref_prim.IsInstanceable():
            print(f"[ERROR] Reference {OBJECT_REF_PATH} should be instanceable!")
        else:
            print(f"[INFO]   ✓ Reference {OBJECT_REF_PATH} is instanceable")

    await omni.kit.app.get_app().next_update_async()

    #Clone envs
    print(f"[INFO] Cloning template env_0 into grid (NUM_COPIES={NUM_COPIES}, spacing={CLONE_SPACING})...")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)

    # /World/Envs/env_0 ... /World/Envs/env_{N-1}
    env_paths = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_COPIES)
    print(f"[DEBUG] Cloner env_paths[0..min]: {env_paths[:min(5, len(env_paths))]}")
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths)

    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    print(f"[INFO] Cloning done: {len(env_paths)} envs")
    # Debug: confirm env transforms differ (so clones are spatially separated)
    try:
        p0, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(0)))
        p1, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(1))) if NUM_COPIES > 1 else (None, None)
        print(f"[DEBUG] env_0 world pos: {p0}")
        if p1 is not None:
            print(f"[DEBUG] env_1 world pos: {p1} (delta ~ {p1 - p0})")
    except Exception as e:
        print(f"[WARN] Could not print env transforms: {e}")
    
    env0_pos, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(0)))
    env_deltas = []
    obj_motion_prims = []
    part_paths = []
    left_finger_paths = []
    right_finger_paths = []

    for k in range(NUM_COPIES):
        pk, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(k)))
        env_deltas.append(pk - env0_pos)

        obj_motion_prims.append(stage.GetPrimAtPath(obj_wrap(k)))      # wrapper (non-instanceable)
        part_paths.append(get_geom_root_path(stage, obj_wrap(k)))      # bbox target
        left_finger_paths.append(f"{grip_ref(k)}/panda_leftfinger")
        right_finger_paths.append(f"{grip_ref(k)}/panda_rightfinger")

    print(f"[INFO] Prepared env deltas + per-env prim paths")
    # -----------------------
    # Per-env managers for env-specific Close joint paths
    # -----------------------
    managers = []
    open_targets_list = []
    close_targets_list = []

    tmp_cfg_dir = output_dir / "_tmp_env_cfgs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    for k in range(NUM_COPIES):
        gm = GraspingManager()

        env_yaml = tmp_cfg_dir / f"grasp_config_env_{k}.yaml"
        shutil.copy(str(yaml_path), str(env_yaml))

        geom_root_k = get_geom_root_path(stage, obj_wrap(k))
        load_and_modify_yaml_inplace(
            env_yaml,
            geom_root_k,
            grip_wrap(k),
            GRASPS_PER_PART,
            obj_wrap(k),
            "WholeObject",
            None,
        )

        if not gm.load_config(str(env_yaml)):
            raise RuntimeError(f"Failed to load env config for env {k}")

        open_phase_k = gm.get_grasp_phase_by_name("Open")
        open_targets_list.append(open_phase_k.joint_drive_targets if open_phase_k else {})

        close_phase_k = gm.get_grasp_phase_by_name("Close")
        close_targets_list.append(close_phase_k.joint_drive_targets if close_phase_k else {})

        gm.store_initial_gripper_pose()
        managers.append(gm)

    print(f"[INFO] Built {len(managers)} per-env managers")
    probe_offsets = np.empty((0, 3), dtype=np.float32)
    
    if has_parts and mesh_samples:
        print(f"\n[INFO] Computing gripper data at runtime (gripper wrapper is non-instanceable)...")
        probe_offsets = compute_probe_offsets_in_gripper_frame(stage, GRIPPER_REF_PATH)
        print(f"[INFO] Gripper data computed:")
        print(f"  - {len(probe_offsets)} probe offsets")
    await ensure_timeline_playing()
    print(f"[INFO] Timeline is now running")
    object_force_state = None

    batch_size = NUM_COPIES

    for base in range(0, len(poses), batch_size):
        batch = poses[base: base + batch_size]
        K = len(batch)

        if base % 50 == 0:
            print(f"      Evaluating grasps {base+1}..{base+K} / {len(poses)} (batch size={K})...")

        try:
            await ensure_timeline_playing()

            # Unpack batch - NOW INCLUDES target_part (3 elements per pose)
            locs, quats, target_parts = [], [], []
            start_locs, start_quats = [], []
            
            for k in range(K):
                loc_k, quat_k, target_part_k = batch[k]  # CHANGED: Unpack 3 elements
                
                locs.append(loc_k)
                quats.append(quat_k)
                target_parts.append(target_part_k)  # NEW: Store target part

                sL, sQ = offset_pose_along_local_z(loc_k, quat_k, -APPROACH_DIST)
                start_locs.append(sL)
                start_quats.append(sQ)
                managers[k].set_gripper_pose(sL, sQ)
                
            # Reset envs used in this batch
            for k in range(K):
                managers[k].clear_simulation(simulate_using_timeline=False)
                stop_gripper_base_env(stage, k)

                ps_prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
                if not ps_prim.IsValid():
                    print("[WARN] Physics scene was removed, recreating...")
                    setup_physics_scene(stage)
                    await omni.kit.app.get_app().next_update_async()

            await omni.kit.app.get_app().next_update_async()
            object_force_state = create_object_force_tensor_view(NUM_COPIES)

            # Record poses BEFORE approach
            init_poses_before_approach = []
            for k in range(K):
                p0, q0 = transform_utils.get_prim_world_pose(obj_motion_prims[k])
                init_poses_before_approach.append((p0, q0))

            # Batched linear approach
            for t in range(MOVE_STEPS + 1):
                alpha = t / float(MOVE_STEPS)
                for k in range(K):
                    cur = (start_locs[k] * (1.0 - alpha)) + (locs[k] * alpha)
                    managers[k].set_gripper_pose(cur, start_quats[k])
                    ot = open_targets_list[k]
                    if ot:
                        apply_joint_targets(stage, ot)
                await omni.kit.app.get_app().next_update_async()

            await step_simulation(10)

            # Gate: approach collision per env
            active = [True] * K
            for k in range(K):
                p1, _ = transform_utils.get_prim_world_pose(obj_motion_prims[k])
                approach_disp = (p1 - init_poses_before_approach[k][0]).GetLength()
                if approach_disp > DISTANCE_THRESHOLD:
                    grasps_rejected_approach_collision += 1
                    active[k] = False

            # Record poses BEFORE closing (for transform computation)
            poses_before_close = []
            for k in range(K):
                p_bc, q_bc = transform_utils.get_prim_world_pose(obj_motion_prims[k])
                poses_before_close.append((p_bc, q_bc))

            # Close gripper
            for _ in range(64):
                for k in range(K):
                    if not active[k]:
                        continue
                    ct = close_targets_list[k]
                    if ct:
                        apply_joint_targets(stage, ct)
                await omni.kit.app.get_app().next_update_async()

            # Record poses AFTER closing (to compute transform)
            poses_after_close = []
            for k in range(K):
                p_ac, q_ac = transform_utils.get_prim_world_pose(obj_motion_prims[k])
                poses_after_close.append((p_ac, q_ac))

            # === POST-CLOSE: DEVIATION + PART DETECTION + PENETRATION ===
            contacted_parts = [None] * K  # Track which part was actually contacted
            object_transforms = [None] * K  # Store (delta_pos, delta_rot) per env (only for has_parts)
            
            for k in range(K):
                if not active[k]:
                    continue

                p_before, q_before = poses_before_close[k]
                p_after, q_after = poses_after_close[k]

                # Check Z-axis deviation
                deg_deviation = get_object_z_axis_deviation(obj_prim, q_before, q_after)
                if deg_deviation > MAX_Z_AXIS_DEVIATION:
                    grasps_rejected_closure += 1
                    active[k] = False
                    continue

                # === PART DETECTION (only if has_parts) ===
                if has_parts and mesh_samples and probe_offsets.shape[0] > 0:
                    # Compute object transform (how much object moved during closure)
                    delta_pos, delta_rot = compute_transform_from_poses(
                        p_before, q_before, p_after, q_after
                    )
                    object_transforms[k] = (delta_pos, delta_rot)
                    
                    # Transform probes by grasp pose (in initial frame)
                    probe_positions_initial = transform_probes_by_grasp_pose(
                        probe_offsets, locs[k], quats[k]
                    )
                    
                    # Apply INVERSE transform to bring probes back to object's frame
                    probe_positions_corrected = np.empty_like(probe_positions_initial)
                    for i in range(len(probe_positions_initial)):
                        probe_positions_corrected[i] = apply_inverse_transform_to_point(
                            probe_positions_initial[i], delta_pos, delta_rot
                        )
                    
                    # Match to mesh samples (which are in initial frame)
                    matched_results = batch_match_probes_to_meshes_local(
                        probe_positions_corrected, mesh_samples
                    )
                    
                    # Find closest probe
                    if matched_results:
                        best_match = min(matched_results, key=lambda x: x[1])
                        best_mesh_path, best_dist = best_match
                        
                        contacted_part = extract_part_from_mesh_path(best_mesh_path, parts_list)
                        
                    else:
                        contacted_part = "body"
                    
                    contacted_parts[k] = contacted_part
                else:
                    contacted_parts[k] = "body"
                
                # === PENETRATION CHECK ===
                if has_parts:

                    contacted_part_path = None
                    check_part_name = contacted_parts[k] if contacted_parts[k] else "body"
                    
                    if parts_list and len(parts_list) > 1 and contacted_parts[k] != "body":
                        # Find the contacted part's path in this env
                        for part_path_template, pname in parts_list:
                            if pname == contacted_parts[k]:
                                # Map template path to this env's path
                                rel_path = part_path_template.replace(OBJECT_REF_PATH, "")
                                contacted_part_path = f"{obj_ref(k)}{rel_path}"
                                break
                        
                        if not contacted_part_path:
                            # Fallback to whole object
                            contacted_part_path = part_paths[k]
                            check_part_name = "WholeObject"
                    else:
                        contacted_part_path = part_paths[k]
                    
                    ok_pen, _ = check_grasp_penetration_depth(
                        stage,
                        part_path=contacted_part_path,
                        part_name=check_part_name,
                        obj_cat=model_cat,
                        left_finger_path=left_finger_paths[k],
                        right_finger_path=right_finger_paths[k],
                    )
                else:
                    # Use simple runtime method for whole object (matches File 1)
                    ok_pen, _ = check_grasp_penetration_depth(
                        stage,
                        part_path=part_paths[k],
                        part_name="WholeObject",
                        obj_cat=model_cat,
                        left_finger_path=left_finger_paths[k],
                        right_finger_path=right_finger_paths[k],
                    )
                
                if not ok_pen:
                    grasps_rejected_shallow += 1
                    active[k] = False
                    continue

            # === GRAVITY + HOLD + RETRIEVAL (unchanged) ===
            ps_prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
            if not ps_prim.IsValid():
                print("[ERROR] No physics scene for gravity test")
                continue

            for k in range(K):
                if active[k]:
                    set_disable_gravity(stage, obj_wrap(k), False)

            for _ in range(5):
                for k in range(K):
                    if not active[k]:
                        continue
                    ct = close_targets_list[k]
                    if ct:
                        apply_joint_targets(stage, ct)
                await omni.kit.app.get_app().next_update_async()

            z0 = [None] * K
            for k in range(K):
                if active[k]:
                    z0[k] = float(transform_utils.get_prim_world_pose(obj_motion_prims[k])[0][2])

            for _ in range(HOLD_TEST_STEPS):
                for k in range(K):
                    if not active[k]:
                        continue
                    ct = close_targets_list[k]
                    if ct:
                        apply_joint_targets(stage, ct)
                await omni.kit.app.get_app().next_update_async()

            z1 = [None] * K
            z_drop = [None] * K
            for k in range(K):
                if active[k]:
                    z1[k] = float(transform_utils.get_prim_world_pose(obj_motion_prims[k])[0][2])
                    z_drop[k] = z0[k] - z1[k]

            retrieval_end_locs = [None] * K
            z_before_retrieval = [None] * K
            for k in range(K):
                if not active[k]:
                    continue
                retrieval_end_locs[k] = offset_pose_along_world_up(stage, locs[k], +RETRIEVAL_DIST)
                z_before_retrieval[k] = float(transform_utils.get_prim_world_pose(obj_motion_prims[k])[0][2])

            for t in range(MOVE_STEPS + 1):
                alpha = t / float(MOVE_STEPS)

                if t < RETRIEVAL_PERTURBATION_DELAY_STEPS:
                    angle_deg = 0.0
                else:
                    effective_alpha = (t - RETRIEVAL_PERTURBATION_DELAY_STEPS) / float(MOVE_STEPS - RETRIEVAL_PERTURBATION_DELAY_STEPS)
                    angle_deg = RETRIEVAL_PERTURBATION_AMP_DEG * math.sin(
                        2.0 * math.pi * RETRIEVAL_PERTURBATION_FREQ * effective_alpha
                    )

                perturb_rot = Gf.Rotation(Gf.Vec3d(0, 1, 0), angle_deg)
                perturb_quat = perturb_rot.GetQuat()

                for k in range(K):
                    if not active[k]:
                        continue
                    cur = (locs[k] * (1.0 - alpha)) + (retrieval_end_locs[k] * alpha)
                    perturbed_quat = quats[k] * Gf.Quatd(
                        perturb_quat.GetReal(),
                        Gf.Vec3d(*perturb_quat.GetImaginary())
                    )
                    managers[k].set_gripper_pose(cur, perturbed_quat)
                    ct = close_targets_list[k]
                    if ct:
                        apply_joint_targets(stage, ct)

                forces = build_simple_retrieval_force_array(
                    active=active,
                    num_copies=NUM_COPIES,
                    mode=RETRIEVAL_FORCE_TEST_MODE,
                    mag=RETRIEVAL_FORCE_MAG,
                    units=RETRIEVAL_FORCE_UNITS,
                    masses=object_force_state["masses"],
                    frame_idx=t,
                    total_frames=MOVE_STEPS + 1,
                )
                object_force_state = apply_retrieval_forces(object_force_state, forces)

                await omni.kit.app.get_app().next_update_async()

            # Restore gravity off always
            for k in range(K):
                try:
                    set_disable_gravity(stage, obj_wrap(k), True)
                except Exception:
                    pass

            # === SAVE RESULTS WITH PART INFO ===
            for k in range(K):
                if not active[k]:
                    continue

                z_after_retrieval = float(transform_utils.get_prim_world_pose(obj_motion_prims[k])[0][2])
                z_drop_retrieval = z_before_retrieval[k] + RETRIEVAL_DIST - z_after_retrieval

                if z_drop[k] > HOLD_FALL_Z_DROP_THRESH or z_drop_retrieval > HOLD_FALL_Z_DROP_THRESH:
                    grasps_rejected_retrieval += 1
                    continue

                loc_t, quat_t, target_part_t = batch[k]  # CHANGED: Unpack target part
                contacted_part_t = contacted_parts[k] if contacted_parts[k] else "body"  # NEW
                
                results.append({
                    "index": len(results),
                    "success": True,
                    "displacement": 0.0,
                    "displacement_passed": True,
                    "distance_check_passed": True,
                    "part": contacted_part_t,  # NEW: Actual contacted part (detected)
                    "target_part": target_part_t,  # NEW: Intended target part (from generation)
                    "pose": {
                        "position": [float(loc_t[0]), float(loc_t[1]), float(loc_t[2])],
                        "orientation": {"w": float(quat_t.GetReal()), "xyz": [float(v) for v in quat_t.GetImaginary()]},
                    },
                })

        except Exception as e:
            print(f"      Error on batch starting at grasp {base}: {e}")
            import traceback
            traceback.print_exc()
            continue
     
    successful = sum(1 for r in results if r['success'])
    print(f"    [INFO] {len(results)} evaluated, {successful} successful")
    print(f"    [INFO] Rejected {grasps_rejected_approach_collision} during approach (collision)")
    print(f"    [INFO] Rejected {grasps_rejected_closure} during close (bad closure or displacement)")
    print(f"    [INFO] Rejected {grasps_rejected_shallow} due to shallow penetration")
    print(f"    [INFO] Rejected {grasps_rejected_retrieval} during retrieval (object didn't follow)")
    
    all_results = results
    
    print(f"\n[INFO] Total grasps: {len(all_results)}")
    print(f"[INFO] Successful: {sum(1 for r in all_results if r['success'])}")
    
    functional_pairs = None
    if FUNCTIONAL_PAIRS_PATH.exists():
        try:
            functional_pairs = load_functional_pairs(FUNCTIONAL_PAIRS_PATH)
            print(f"[INFO] Loaded {len(functional_pairs)} functional pairs")
        except Exception as e:
            print(f"[WARN] Failed to load functional pairs: {e}")
    
    functional_grasp, nonfunctional_grasp, stats = classify_grasps(all_results, model_cat, functional_pairs)

    # If no functional grasps were assigned (e.g. body-only object with no functional pairs),
    # promote all non-functional grasps into functional_grasp so the output is never empty.
    if not functional_grasp and nonfunctional_grasp:
        print(f"[INFO] No functional_grasp entries — promoting all body grasps to functional_grasp")
        functional_grasp = nonfunctional_grasp
        nonfunctional_grasp = {}

    print(f"\n[INFO] Classification:")
    print(f"  Ignored (new/original): {stats['grasps_ignored']}")
    print(f"  Functional parts: {stats['functional_parts']}")
    print(f"  Functional grasps: {stats['total_functional_grasps']}")
    print(f"  Non-functional parts: {stats['nonfunctional_parts']}")
    print(f"  Non-functional grasps: {stats['total_nonfunctional_grasps']}")

    successful_results = [r for r in all_results if r["success"]]
    out_json_success = output_dir / "grasp_results_successful.json"
    out_json_success.write_text(json.dumps(successful_results, indent=2))
    print(f"\n✓ Saved: {out_json_success}")

    if functional_pairs is not None and (functional_grasp or nonfunctional_grasp):
        classified_output = {
            "type": model_cat,
            "bottom_center": bottom_center_list,
            "functional_grasp": functional_grasp,
            "grasp": nonfunctional_grasp
        }
        out_json_classified = output_dir / "grasp_pose.json"
        out_json_classified.write_text(json.dumps(classified_output, indent=2))
        print(f"✓ Saved: {out_json_classified}")
    
    grasping_manager.clear()

    print(f"[INFO] Stopping timeline before cleanup...")
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()

    try:
        if yaml_path.exists():
            yaml_path.unlink()
            print(f"[INFO] Deleted YAML config: {yaml_path}")
        if out_json_success.exists():
            out_json_success.unlink()
            print(f"[INFO] Deleted intermediate results: {out_json_success}")
        tmp_cfg_dir = output_dir / "_tmp_env_cfgs"
        if tmp_cfg_dir.exists():
            shutil.rmtree(tmp_cfg_dir)
            
        annotation_dir = output_dir / "Annotation"
        annotation_dir.mkdir(parents=True, exist_ok=True)

        classified_src = output_dir / "grasp_pose.json"
        classified_dst = annotation_dir / "grasp_pose.json"

        if classified_src.exists():
            shutil.move(str(classified_src), str(classified_dst))
            print(f"[INFO] Moved classified annotation to: {classified_dst}")
        else:
            print(f"[INFO] No classified annotation JSON found for {obj_id}; nothing to move.")

    except Exception as e:
        print(f"[WARN] Post-processing cleanup failed for {obj_id}: {e}")

    print(f"\n{'='*80}")
    print(f"Completed: {obj_id}")
    print(f"{'='*80}\n")

def add_reference(ref_path: str, usd_file: Path):
    object_prim = add_reference_to_stage(usd_path=str(usd_file), prim_path=ref_path)
    if not object_prim:
        print(f"[ERROR] Failed to add reference at {ref_path}")
        return None
    print(f"[INFO] Loaded {usd_file.name} at {ref_path}")
    return object_prim

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

def load_and_modify_yaml_inplace(yaml_path: Path, object_path: str, gripper_path: str, num_candidates: int, part_path: str = "", part_name: str = "", grasp_axis_override=None):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # --- Remap any joint paths that were authored for a different env/gripper ---
    # Many configs store joint paths under <gripper_path>/ref/... . When we clone envs,
    # the wrapper path changes (e.g., /World/Envs/env_0/Flying_hand_probe -> env_19),
    # so we must rewrite keys in joint_pregrasp_states and grasp_phases[*].joint_drive_targets.
    old_gripper_path = data.get("gripper_path", "")
    old_ref_prefix = f"{old_gripper_path}/ref" if old_gripper_path else ""
    new_ref_prefix = f"{gripper_path}/ref"

    def _remap_dict_keys(d: dict) -> dict:
        if not isinstance(d, dict) or not d:
            return d
        out = {}
        for k, v in d.items():
            if isinstance(k, str) and old_ref_prefix and k.startswith(old_ref_prefix):
                nk = k.replace(old_ref_prefix, new_ref_prefix, 1)
            else:
                nk = k
            out[nk] = v
        return out

    # Remap pregrasp joints
    if "joint_pregrasp_states" in data:
        data["joint_pregrasp_states"] = _remap_dict_keys(data.get("joint_pregrasp_states", {}))

    # Remap per-phase joint targets
    phases = data.get("grasp_phases", [])
    if isinstance(phases, list):
        for ph in phases:
            if not isinstance(ph, dict):
                continue
            if "joint_drive_targets" in ph:
                ph["joint_drive_targets"] = _remap_dict_keys(ph.get("joint_drive_targets", {}))

    # --- Update object/gripper paths for this env ---
    data["object_path"] = object_path
    data["gripper_path"] = gripper_path

    if "sampler_config" not in data or not isinstance(data["sampler_config"], dict):
        data["sampler_config"] = {}

    data["sampler_config"]["num_candidates"] = int(num_candidates)

    if grasp_axis_override:
        data["sampler_config"]["grasp_align_axis"] = grasp_axis_override
        data["sampler_config"]["orientation_sample_axis"] = grasp_axis_override
    else:
        data["sampler_config"]["grasp_align_axis"] = [0, 1, 0]
        data["sampler_config"]["orientation_sample_axis"] = [0, 1, 0]

    data["num_orientations"] = 4

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"    [DEBUG] YAML updated → object={object_path}, gripper={gripper_path}, grasps={num_candidates}")

def offset_pose_along_local_z(position, quaternion, offset: float) -> tuple:
    rot = Gf.Rotation(quaternion)
    local_z_axis = Gf.Vec3d(0, 0, 1)
    world_z_direction = rot.TransformDir(local_z_axis)
    world_offset = world_z_direction.GetNormalized() * offset
    new_position = position + world_offset
    
    return new_position, quaternion

def offset_pose_along_world_up(stage, position, offset: float):
    up = UsdGeom.GetStageUpAxis(stage)
    if up == UsdGeom.Tokens.z:
        world_up = Gf.Vec3d(0, 0, 1)
    elif up == UsdGeom.Tokens.y:
        world_up = Gf.Vec3d(0, 1, 0)
    else:
        world_up = Gf.Vec3d(0, 0, 1)

    return position + world_up * offset

async def step_simulation(steps=1):
    for _ in range(steps):
        await omni.kit.app.get_app().next_update_async()

def get_interpolated_poses(start_pos, end_pos, steps: int):
    path = []
    for i in range(steps + 1):
        alpha = i / float(steps)
        interp_pos = (start_pos * (1.0 - alpha)) + (end_pos * alpha)
        path.append(interp_pos)
    return path

def apply_joint_targets(stage, joint_targets: dict):
    for joint_path, target_val in joint_targets.items():
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
            
        props = prim.GetProperties()
        for prop in props:
            name = prop.GetName()
            if "drive" in name and "targetPosition" in name:
                prop.Set(float(target_val))

def _get_or_create_attr(prim, name: str, sdf_type):
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, sdf_type)
    return attr

def set_physics_linear_velocity(stage, prim_path: str, v: Gf.Vec3d):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for velocity: {prim_path}")
    attr = _get_or_create_attr(prim, "physics:velocity", Sdf.ValueTypeNames.Vector3f)
    attr.Set(Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])))

def set_physics_angular_velocity(stage, prim_path: str, w: Gf.Vec3d):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for angular velocity: {prim_path}")
    attr = _get_or_create_attr(prim, "physics:angularVelocity", Sdf.ValueTypeNames.Vector3f)
    attr.Set(Gf.Vec3f(float(w[0]), float(w[1]), float(w[2])))

def stop_gripper_base(stage):
    set_physics_linear_velocity(stage, GRIPPER_BASE_PATH, Gf.Vec3d(0, 0, 0))
    set_physics_angular_velocity(stage, GRIPPER_BASE_PATH, Gf.Vec3d(0, 0, 0))

def stop_gripper_base_env(stage, k: int):
    global GRIPPER_BASE_PATH
    old = GRIPPER_BASE_PATH
    try:
        GRIPPER_BASE_PATH = grip_base(k)
        stop_gripper_base(stage)
    finally:
        GRIPPER_BASE_PATH = old

async def simulate_linear_movement(grasping_manager, start_pos, start_rot, end_pos, steps):
    with suppress_stdout():
        open_phase = grasping_manager.get_grasp_phase_by_name("Open")
    
    if not open_phase:
        print("[ERROR] No 'Open' phase found - falling back to basic movement")
        for i in range(steps + 1):
            alpha = i / float(steps)
            current_pos = (start_pos * (1.0 - alpha)) + (end_pos * alpha)
            grasping_manager.set_gripper_pose(current_pos, start_rot)
            await omni.kit.app.get_app().next_update_async()
        return
    
    original_steps = open_phase.simulation_steps
    open_phase.simulation_steps = 1
    
    try:
        for i in range(steps + 1):
            alpha = i / float(steps)
            current_pos = (start_pos * (1.0 - alpha)) + (end_pos * alpha)
            
            grasping_manager.set_gripper_pose(current_pos, start_rot)
            
            with suppress_stdout():
                await grasping_manager.simulate_single_grasp_phase(
                    phase_identifier="Open",
                    render=True,
                    simulate_using_timeline=True
                )
    finally:
        open_phase.simulation_steps = original_steps

async def retrieval_with_pose(grasping_manager, stage, start_pos, start_rot, end_pos, steps, close_targets, settle_per_step=1):
    for i in range(steps + 1):
        alpha = i / float(steps)
        cur = (start_pos * (1.0 - alpha)) + (end_pos * alpha)

        grasping_manager.set_gripper_pose(cur, start_rot)

        if close_targets:
            apply_joint_targets(stage, close_targets)

        for _ in range(max(1, int(settle_per_step))):
            await omni.kit.app.get_app().next_update_async()

def filter_bottom_up_grasps(poses):
    filtered_poses = []
    
    for loc, quat in poses:
        rot = Gf.Rotation(quat)
        local_z = Gf.Vec3d(0, 0, 1)
        world_z_direction = rot.TransformDir(local_z)
        world_z_direction = world_z_direction.GetNormalized()
        
        if world_z_direction[2] <= 0:
            filtered_poses.append((loc, quat))
    
    return filtered_poses

def get_object_z_axis_deviation(obj_prim, init_orientation, final_orientation) -> float:
    init_rot = Gf.Rotation(init_orientation)
    final_rot = Gf.Rotation(final_orientation)
    
    local_z = Gf.Vec3d(0, 0, 1)
    
    init_z_world = init_rot.TransformDir(local_z)
    final_z_world = final_rot.TransformDir(local_z)
    
    init_z_world = init_z_world.GetNormalized()
    final_z_world = final_z_world.GetNormalized()
    
    dot_product = init_z_world * final_z_world
    dot_product = max(-1.0, min(1.0, dot_product))
    
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    
    return float(angle_degrees)

def check_grasp_penetration_depth(
    stage,
    part_path: str,
    part_name: str,
    obj_cat: str,
    left_finger_path: str | None = None,
    right_finger_path: str | None = None,
    min_horizontal_penetration_percent: float = None,
    min_z_penetration_percent: float = None
) -> tuple[bool, dict]:
    
    if min_horizontal_penetration_percent is None or min_z_penetration_percent is None:
        min_horizontal_penetration_percent, min_z_penetration_percent = get_penetration_thresholds(
            obj_cat, part_name
        )
        print(f"      [DEBUG] Using thresholds for {obj_cat}/{part_name}: H={min_horizontal_penetration_percent}%, Z={min_z_penetration_percent}%")
    
    if left_finger_path is None:
        left_finger_path = f"{GRIPPER_REF_PATH}/panda_leftfinger"
    if right_finger_path is None:
        right_finger_path = f"{GRIPPER_REF_PATH}/panda_rightfinger"
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    
    part_prim = stage.GetPrimAtPath(part_path)
    left_finger_prim = stage.GetPrimAtPath(left_finger_path)
    right_finger_prim = stage.GetPrimAtPath(right_finger_path)
    
    if not all([part_prim.IsValid(), left_finger_prim.IsValid(), right_finger_prim.IsValid()]):
        return True, {}
    
    part_bbox = bbox_cache.ComputeWorldBound(part_prim)
    left_bbox = bbox_cache.ComputeWorldBound(left_finger_prim)
    right_bbox = bbox_cache.ComputeWorldBound(right_finger_prim)
    
    part_range = part_bbox.GetRange()
    left_range = left_bbox.GetRange()
    right_range = right_bbox.GetRange()
    
    part_min, part_max = part_range.GetMin(), part_range.GetMax()
    left_min, left_max = left_range.GetMin(), left_range.GetMax()
    right_min, right_max = right_range.GetMin(), right_range.GetMax()
    
    object_width_x = part_max[0] - part_min[0]
    object_width_y = part_max[1] - part_min[1]
    object_height_z = part_max[2] - part_min[2]
    
    if object_height_z < 0.001:
        object_height_z = 0.001
    if object_width_x < 0.001:
        object_width_x = 0.001
    if object_width_y < 0.001:
        object_width_y = 0.001
    
    max_object_xy_dimension = max(object_width_x, object_width_y)
    
    grasp_box_min = Gf.Vec3d(
        min(left_min[0], right_min[0]),
        min(left_min[1], right_min[1]),
        min(left_min[2], right_min[2])
    )
    grasp_box_max = Gf.Vec3d(
        max(left_max[0], right_max[0]),
        max(left_max[1], right_max[1]),
        max(left_max[2], right_max[2])
    )
    
    overlap_x = max(0, min(grasp_box_max[0], part_max[0]) - max(grasp_box_min[0], part_min[0]))
    overlap_y = max(0, min(grasp_box_max[1], part_max[1]) - max(grasp_box_min[1], part_min[1]))
    overlap_z = max(0, min(grasp_box_max[2], part_max[2]) - max(grasp_box_min[2], part_min[2]))
    
    horizontal_overlap = max(overlap_x, overlap_y)
    horizontal_percent = (horizontal_overlap / max_object_xy_dimension) * 100.0
    z_percent = (overlap_z / object_height_z) * 100.0
    
    part_lower = part_name.lower()
    handle_keywords = ['handle', 'grip', 'knob', 'pull']
    is_handle = any(keyword in part_lower for keyword in handle_keywords)
    
    if is_handle:
        is_valid = horizontal_percent >= min_horizontal_penetration_percent
        part_type = "HANDLE"
        check_applied = f"XY only (≥{min_horizontal_penetration_percent}%)"
    else:
        is_valid = z_percent >= min_z_penetration_percent
        part_type = "BODY"
        check_applied = f"Z only (≥{min_z_penetration_percent}%)"
    
    metrics = {
        "horizontal_percent": float(horizontal_percent),
        "z_percent": float(z_percent),
        "part_type": part_type,
        "is_handle": is_handle,
        "check_applied": check_applied,
        "object_dimensions": {
            "width_x": float(object_width_x),
            "width_y": float(object_width_y),
            "height_z": float(object_height_z)
        }
    }
    
    if not is_valid:
        if is_handle:
            print(f"      [DEBUG] HANDLE insufficient XY penetration: {horizontal_percent:.1f}% < {min_horizontal_penetration_percent}%")
        else:
            print(f"      [DEBUG] BODY insufficient Z penetration: {z_percent:.1f}% < {min_z_penetration_percent}%")
    
    return is_valid, metrics

def classify_grasps(all_results: list, model_cat: str, functional_pairs: set | None):
    functional_grasp = defaultdict(list)
    nonfunctional_grasp = defaultdict(list)
    grasps_ignored = 0
    
    for rec in all_results:
        if not rec.get("success", False):
            continue
        
        part = rec.get("part")
        if part is None:
            continue
        
        if should_ignore_part(part):
            grasps_ignored += 1
            continue
        
        try:
            dof7 = extract_pose7(rec["pose"])
        except Exception:
            continue
        
        if functional_pairs is not None:
            if (model_cat, part) in functional_pairs:
                functional_grasp[part].append(dof7)
            else:
                nonfunctional_grasp[part].append(dof7)
    
    stats = {
        "grasps_ignored": grasps_ignored,
        "functional_parts": len(functional_grasp),
        "nonfunctional_parts": len(nonfunctional_grasp),
        "total_functional_grasps": sum(len(v) for v in functional_grasp.values()),
        "total_nonfunctional_grasps": sum(len(v) for v in nonfunctional_grasp.values()),
    }
    
    return dict(functional_grasp), dict(nonfunctional_grasp), stats

# =======================
# Entry Point
# =======================
def main():
    global simulation_app
    
    print("[INFO] Starting grasp generation pipeline...")
    
    try:
        loop = asyncio.get_event_loop()
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

if __name__ == "__main__":
    main()
