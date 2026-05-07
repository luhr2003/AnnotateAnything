from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt


@dataclass
class ObjectMeshData:
    vertices: np.ndarray
    faces: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignedDistanceField:
    sdf_grid: np.ndarray
    gradient_grid: np.ndarray
    origin: np.ndarray
    voxel_size: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.sdf_grid.shape)

    def save(self, path: Path) -> Path:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(_json_safe(self.metadata), sort_keys=True)
        np.savez_compressed(
            str(path),
            sdf_grid=self.sdf_grid.astype(np.float32),
            gradient_grid=self.gradient_grid.astype(np.float32),
            origin=self.origin.astype(np.float32),
            voxel_size=np.asarray([self.voxel_size], dtype=np.float32),
            metadata_json=np.asarray([metadata_json]),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SignedDistanceField":
        with np.load(str(path), allow_pickle=True) as raw:
            metadata: Dict[str, Any] = {}
            if "metadata_json" in raw and len(raw["metadata_json"]) > 0:
                metadata_str = str(np.asarray(raw["metadata_json"]).reshape(-1)[0])
                if metadata_str:
                    metadata = dict(json.loads(metadata_str))
            elif "metadata" in raw:
                try:
                    metadata_array = raw["metadata"]
                    if len(metadata_array) > 0:
                        metadata = dict(metadata_array[0])
                except Exception:
                    # Older caches stored metadata as a pickled object array. If that
                    # pickle is no longer readable under the current NumPy build, the
                    # SDF grids are still usable, so we drop metadata and continue.
                    metadata = {}
            return cls(
                sdf_grid=np.asarray(raw["sdf_grid"], dtype=np.float64),
                gradient_grid=np.asarray(raw["gradient_grid"], dtype=np.float64),
                origin=np.asarray(raw["origin"], dtype=np.float64),
                voxel_size=float(np.asarray(raw["voxel_size"]).reshape(-1)[0]),
                metadata=metadata,
            )

    def signed_distance(self, points_world: np.ndarray) -> np.ndarray:
        return _sample_grid_trilinear(
            self.sdf_grid,
            origin=self.origin,
            voxel_size=self.voxel_size,
            points_world=points_world,
        )

    def gradient(self, points_world: np.ndarray) -> np.ndarray:
        grads: List[np.ndarray] = []
        for axis in range(3):
            grads.append(
                _sample_grid_trilinear(
                    self.gradient_grid[axis],
                    origin=self.origin,
                    voxel_size=self.voxel_size,
                    points_world=points_world,
                )
            )
        return np.stack(grads, axis=1)

    def project_to_surface(self, points_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64)
        sdf = self.signed_distance(pts)
        grad = self.gradient(pts)
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_safe = np.where(grad_norm > 1e-8, grad / grad_norm, 0.0)
        return pts - grad_safe * sdf[:, None]


class ObjectQueryBackend:
    def __init__(self, sdf: SignedDistanceField) -> None:
        self.sdf = sdf

    def signed_distance(self, points_world: np.ndarray) -> np.ndarray:
        return self.sdf.signed_distance(points_world)

    def gradient(self, points_world: np.ndarray) -> np.ndarray:
        return self.sdf.gradient(points_world)

    def project_to_surface(self, points_world: np.ndarray) -> np.ndarray:
        return self.sdf.project_to_surface(points_world)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sample_grid_trilinear(
    grid: np.ndarray,
    *,
    origin: np.ndarray,
    voxel_size: float,
    points_world: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]

    shape = np.asarray(grid.shape, dtype=np.int64)
    coords = (pts - origin[None, :]) / float(voxel_size)

    coords_clipped = np.clip(coords, 0.0, shape[None, :] - 1.0)
    outside_delta = coords - coords_clipped
    outside_distance = np.linalg.norm(outside_delta * float(voxel_size), axis=1)

    i0 = np.floor(coords_clipped).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, shape[None, :] - 1)
    frac = coords_clipped - i0.astype(np.float64)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

    c000 = grid[x0, y0, z0]
    c001 = grid[x0, y0, z1]
    c010 = grid[x0, y1, z0]
    c011 = grid[x0, y1, z1]
    c100 = grid[x1, y0, z0]
    c101 = grid[x1, y0, z1]
    c110 = grid[x1, y1, z0]
    c111 = grid[x1, y1, z1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    samples = c0 * (1.0 - fz) + c1 * fz

    is_outside = np.any((coords < 0.0) | (coords > (shape[None, :] - 1.0)), axis=1)
    samples = np.asarray(samples, dtype=np.float64)
    samples[is_outside] = np.maximum(samples[is_outside], 0.0) + outside_distance[is_outside]
    return samples


def default_sdf_cache_path(usd_path: Path) -> Path:
    usd_path = usd_path.resolve()
    if usd_path.stem.lower() == "object":
        return usd_path.parent / "sdf" / "object_sdf.npz"
    return usd_path.parent / "sdf" / f"{usd_path.stem}_sdf.npz"


def _gf_matrix_to_np(mat) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = mat[i][j]
    return out.T


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ hom.T).T
    return out[:, :3]


def _fan_triangulate(face_counts: Sequence[int], face_indices: Sequence[int]) -> np.ndarray:
    counts = list(face_counts)
    indices = list(face_indices)
    tris: List[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        if count < 3:
            cursor += count
            continue
        poly = indices[cursor : cursor + count]
        cursor += count
        for i in range(1, count - 1):
            tris.append((poly[0], poly[i], poly[i + 1]))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def ensure_object_reference_in_stage(
    usd_path: Path,
    *,
    prim_path: str = "/World/Object",
):
    """
    Add a USD object to the current Isaac Sim stage if that environment is available.
    """
    try:
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
    except Exception as exc:  # pragma: no cover - depends on Isaac Sim runtime
        raise RuntimeError(
            "Isaac Sim stage APIs are unavailable. Run this inside an Isaac Sim Python environment."
        ) from exc

    stage = omni.usd.get_context().get_stage()
    prim = add_reference_to_stage(
        usd_path=str(Path(usd_path).resolve()),
        prim_path=prim_path,
    )
    return stage, prim


def _collect_mesh_prims_from_stage(
    stage,
    *,
    root_prim_path: Optional[str],
    exclude_prim_paths: Optional[Iterable[str]] = None,
):
    from pxr import UsdGeom

    if root_prim_path:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim or not root_prim.IsValid():
            raise ValueError(f"Invalid root prim path: {root_prim_path}")
    else:
        root_prim = stage.GetDefaultPrim()
        if not root_prim or not root_prim.IsValid():
            children = list(stage.GetPseudoRoot().GetChildren())
            if not children:
                raise RuntimeError("USD stage has no traversable root children.")
            root_prim = children[0]

    exclude = {str(x) for x in (exclude_prim_paths or [])}
    mesh_prims = []
    for prim in UsdGeom.Imageable(root_prim).GetPrim().GetStage().Traverse():
        prim_path = str(prim.GetPrimPath())
        if not prim_path.startswith(str(root_prim.GetPrimPath())):
            continue
        if prim_path in exclude:
            continue
        if prim.GetTypeName() == "Mesh":
            mesh_prims.append(prim)
    return mesh_prims


def _assemble_mesh_from_mesh_prims(mesh_prims) -> ObjectMeshData:
    from pxr import UsdGeom

    vertices_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    vertex_offset = 0

    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if points_attr is None or counts_attr is None or indices_attr is None:
            continue

        points = np.asarray(points_attr, dtype=np.float64)
        if len(points) == 0:
            continue

        faces = _fan_triangulate(counts_attr, indices_attr)
        if len(faces) == 0:
            continue

        xform = UsdGeom.Xformable(prim)
        world_T = _gf_matrix_to_np(xform.ComputeLocalToWorldTransform(0.0))
        points_world = _transform_points(points, world_T)

        vertices_all.append(points_world)
        faces_all.append(faces + vertex_offset)
        vertex_offset += points_world.shape[0]

    if not vertices_all:
        raise RuntimeError("No mesh geometry found to assemble into an SDF.")

    vertices = np.concatenate(vertices_all, axis=0)
    faces = np.concatenate(faces_all, axis=0)
    return ObjectMeshData(
        vertices=vertices,
        faces=faces,
        bbox_min=vertices.min(axis=0),
        bbox_max=vertices.max(axis=0),
    )


def load_object_mesh_from_usd(
    usd_path: Path,
    *,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[Iterable[str]] = None,
) -> ObjectMeshData:
    try:
        from pxr import Usd
    except Exception as exc:  # pragma: no cover - depends on USD runtime
        raise RuntimeError(
            "USD APIs are unavailable. Build or load the SDF cache from an Isaac Sim / USD Python environment."
        ) from exc

    stage = Usd.Stage.Open(str(Path(usd_path).resolve()))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    mesh_prims = _collect_mesh_prims_from_stage(
        stage,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    return _assemble_mesh_from_mesh_prims(mesh_prims)


def load_object_mesh_from_stage(
    *,
    root_prim_path: str,
    exclude_prim_paths: Optional[Iterable[str]] = None,
) -> ObjectMeshData:
    try:
        import omni.usd
    except Exception as exc:  # pragma: no cover - depends on Isaac Sim runtime
        raise RuntimeError(
            "Isaac Sim stage APIs are unavailable. Run this inside an Isaac Sim Python environment."
        ) from exc

    stage = omni.usd.get_context().get_stage()
    mesh_prims = _collect_mesh_prims_from_stage(
        stage,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    return _assemble_mesh_from_mesh_prims(mesh_prims)


def build_sdf_from_mesh(
    mesh_data: ObjectMeshData,
    *,
    voxel_size: float,
    padding_voxels: int = 8,
) -> SignedDistanceField:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_data.vertices, dtype=np.float64),
        faces=np.asarray(mesh_data.faces, dtype=np.int64),
        process=False,
    )
    if not mesh.is_watertight:
        # Keep the pipeline usable for imperfect assets, but record the risk.
        watertight = False
    else:
        watertight = True

    voxelized = mesh.voxelized(pitch=float(voxel_size)).fill()
    occupied = np.asarray(voxelized.matrix, dtype=bool)
    if int(padding_voxels) > 0:
        occupied = np.pad(
            occupied,
            pad_width=int(padding_voxels),
            mode="constant",
            constant_values=False,
        )

    dist_outside = distance_transform_edt(~occupied) * float(voxel_size)
    dist_inside = distance_transform_edt(occupied) * float(voxel_size)
    sdf_grid = dist_outside - dist_inside

    gradients = np.stack(
        np.gradient(
            sdf_grid,
            float(voxel_size),
            float(voxel_size),
            float(voxel_size),
            edge_order=1,
        ),
        axis=0,
    )

    origin = np.asarray(voxelized.transform[:3, 3], dtype=np.float64)
    origin = origin - float(padding_voxels) * float(voxel_size)
    return SignedDistanceField(
        sdf_grid=np.asarray(sdf_grid, dtype=np.float64),
        gradient_grid=np.asarray(gradients, dtype=np.float64),
        origin=origin,
        voxel_size=float(voxel_size),
        metadata={
            **mesh_data.metadata,
            "padding_voxels": int(padding_voxels),
            "watertight": bool(watertight),
            "bbox_min": np.asarray(mesh_data.bbox_min, dtype=np.float64).tolist(),
            "bbox_max": np.asarray(mesh_data.bbox_max, dtype=np.float64).tolist(),
        },
    )


def build_sdf_from_usd(
    usd_path: Path,
    *,
    voxel_size: float,
    padding_voxels: int = 8,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[Iterable[str]] = None,
) -> SignedDistanceField:
    mesh_data = load_object_mesh_from_usd(
        usd_path=usd_path,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    mesh_data.metadata.update(
        {
            "source_usd": str(Path(usd_path).resolve()),
            "root_prim_path": root_prim_path,
        }
    )
    return build_sdf_from_mesh(
        mesh_data,
        voxel_size=voxel_size,
        padding_voxels=padding_voxels,
    )


def build_sdf_from_stage(
    *,
    root_prim_path: str,
    voxel_size: float,
    padding_voxels: int = 8,
    exclude_prim_paths: Optional[Iterable[str]] = None,
) -> SignedDistanceField:
    mesh_data = load_object_mesh_from_stage(
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    mesh_data.metadata.update(
        {
            "source_stage_root_prim": str(root_prim_path),
        }
    )
    return build_sdf_from_mesh(
        mesh_data,
        voxel_size=voxel_size,
        padding_voxels=padding_voxels,
    )


def load_or_build_sdf_from_usd(
    usd_path: Path,
    *,
    cache_path: Optional[Path] = None,
    voxel_size: float,
    padding_voxels: int = 8,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[Iterable[str]] = None,
    force_rebuild: bool = False,
) -> SignedDistanceField:
    cache_path = cache_path.resolve() if cache_path is not None else default_sdf_cache_path(Path(usd_path))
    if cache_path.exists() and not force_rebuild:
        try:
            sdf = SignedDistanceField.load(cache_path)
            sdf.metadata.setdefault("cache_path", str(cache_path))
            sdf.metadata.setdefault("source_usd", str(Path(usd_path).resolve()))
            if root_prim_path is not None:
                sdf.metadata.setdefault("root_prim_path", root_prim_path)
            return sdf
        except Exception as exc:
            print(
                f"[object_query] Rebuilding unreadable SDF cache at {cache_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                cache_path.unlink()
            except OSError:
                pass

    sdf = build_sdf_from_usd(
        usd_path=usd_path,
        voxel_size=voxel_size,
        padding_voxels=padding_voxels,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    sdf.metadata["cache_path"] = str(cache_path)
    sdf.save(cache_path)
    return sdf


def load_or_build_sdf_from_stage(
    *,
    root_prim_path: str,
    cache_path: Path,
    voxel_size: float,
    padding_voxels: int = 8,
    exclude_prim_paths: Optional[Iterable[str]] = None,
    force_rebuild: bool = False,
) -> SignedDistanceField:
    cache_path = Path(cache_path).resolve()
    if cache_path.exists() and not force_rebuild:
        try:
            sdf = SignedDistanceField.load(cache_path)
            sdf.metadata.setdefault("cache_path", str(cache_path))
            sdf.metadata.setdefault("source_stage_root_prim", str(root_prim_path))
            return sdf
        except Exception as exc:
            print(
                f"[object_query] Rebuilding unreadable SDF cache at {cache_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                cache_path.unlink()
            except OSError:
                pass

    sdf = build_sdf_from_stage(
        root_prim_path=root_prim_path,
        voxel_size=voxel_size,
        padding_voxels=padding_voxels,
        exclude_prim_paths=exclude_prim_paths,
    )
    sdf.metadata["cache_path"] = str(cache_path)
    sdf.save(cache_path)
    return sdf


def make_object_query_from_sdf(
    sdf_or_path: SignedDistanceField | Path,
) -> ObjectQueryBackend:
    if isinstance(sdf_or_path, SignedDistanceField):
        sdf = sdf_or_path
    else:
        sdf = SignedDistanceField.load(Path(sdf_or_path))
    return ObjectQueryBackend(sdf)
