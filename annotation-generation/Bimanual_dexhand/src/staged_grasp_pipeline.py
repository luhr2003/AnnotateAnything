from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.contact_resolution import (
    ContactResolutionResult,
    resolve_contact_templates_for_anchors,
    summarize_contact_resolution,
)
from src.hand_kinematics import load_hand_kinematics_model
from src.object_query import (
    default_sdf_cache_path,
    load_or_build_sdf_from_usd,
    make_object_query_from_sdf,
)
from src.optimizer_core import (
    StagedGraspResult,
    optimize_staged_grasp,
    optimize_staged_grasps,
    sort_staged_grasp_results,
    summarize_staged_grasp_result,
)
from src.seed_generation import (
    SeedGenerationResult,
    generate_pose_seeds_for_contacts,
    summarize_seed_generation,
)
from src.types_config import Category, Side

if TYPE_CHECKING:
    from src.pipeline_single_hand import SingleHandProposalResult


@dataclass
class ContactSeedOptimizationBundle:
    contact_result: ContactResolutionResult
    seed_result: SeedGenerationResult
    optimized_results: List[StagedGraspResult]


@dataclass
class SingleHandStagedPipelineResult:
    proposal_result: SingleHandProposalResult
    bundles: List[ContactSeedOptimizationBundle]
    sdf_cache_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


_PROCESS_SINGLE_HAND_CONTEXT: Dict[str, Any] = {}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _runtime_cfg_with_optimizer_overrides(runtime_cfg, optimizer_cfg_overrides: Optional[Dict[str, Any]]):
    if not optimizer_cfg_overrides:
        return runtime_cfg
    cloned = deepcopy(runtime_cfg)
    optimizer_cfg = deepcopy(cloned.optimizer_cfg or {})
    for scope in ("global", "category"):
        override_scope = optimizer_cfg_overrides.get(scope)
        if isinstance(override_scope, dict):
            optimizer_cfg[scope] = _deep_merge_dict(
                optimizer_cfg.get(scope, {}),
                override_scope,
            )
    cloned.optimizer_cfg = optimizer_cfg
    return cloned


def _init_process_candidate_worker(
    runtime_cfg,
    contact_results,
    candidate_seed_lists,
    object_query,
    hand_model,
) -> None:
    global _PROCESS_SINGLE_HAND_CONTEXT
    _PROCESS_SINGLE_HAND_CONTEXT = {
        "runtime_cfg": runtime_cfg,
        "contact_results": contact_results,
        "candidate_seed_lists": candidate_seed_lists,
        "object_query": object_query,
        "hand_model": hand_model,
    }


def _compute_candidate_process(task: tuple[int, int]):
    bundle_idx, seed_idx = task
    context = _PROCESS_SINGLE_HAND_CONTEXT
    result = optimize_staged_grasp(
        runtime_cfg=context["runtime_cfg"],
        contact_result=context["contact_results"][bundle_idx],
        seed=context["candidate_seed_lists"][bundle_idx][seed_idx],
        object_query=context["object_query"],
        hand_model=context["hand_model"],
    )
    return bundle_idx, seed_idx, result


def _effective_candidate_workers(
    *,
    requested_max_workers: int,
    num_candidates_total: int,
) -> int:
    if num_candidates_total <= 0:
        return 0
    if requested_max_workers <= 0:
        return min(num_candidates_total, max(1, os.cpu_count() or 1), 16)
    return min(num_candidates_total, max(1, requested_max_workers))


def run_single_hand_staged(
    *,
    hand_dir: Path,
    global_config_dir: Path,
    side: Side,
    category: Category,
    object_usd: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
    num_surface_points: int = 100000,
    seed: int = 0,
    top_k_anchors: int = 5,
    num_seeds_per_contact: Optional[int] = None,
    top_k_seeds_per_contact: int = 3,
    top_k_optimized_per_contact: Optional[int] = None,
    max_workers: Optional[int] = None,
    sdf_cache_path: Optional[Path] = None,
    force_rebuild_sdf: bool = False,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
    optimizer_cfg_overrides: Optional[Dict[str, Any]] = None,
    parallel_backend: str = "thread",
) -> SingleHandStagedPipelineResult:
    from src.pipeline_single_hand import order_anchors_for_consumption, run_single_hand_region_proposal

    proposal_result = run_single_hand_region_proposal(
        hand_dir=hand_dir,
        global_config_dir=global_config_dir,
        side=side,
        category=category,
        object_usd=object_usd,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        num_surface_points=num_surface_points,
        seed=seed,
    )

    runtime_cfg = _runtime_cfg_with_optimizer_overrides(
        proposal_result.runtime_cfg,
        optimizer_cfg_overrides,
    )
    optimizer_global_cfg = runtime_cfg.optimizer_cfg.get("global", {})
    sdf_cfg = optimizer_global_cfg.get("sdf", {})
    voxel_size = float(sdf_cfg.get("voxel_size", 0.005))
    padding_voxels = int(sdf_cfg.get("padding_voxels", 8))
    force_rebuild_sdf = bool(force_rebuild_sdf or sdf_cfg.get("force_rebuild", False))

    sdf = load_or_build_sdf_from_usd(
        usd_path=object_usd,
        cache_path=sdf_cache_path,
        voxel_size=voxel_size,
        padding_voxels=padding_voxels,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        force_rebuild=force_rebuild_sdf,
    )
    effective_sdf_cache_path = Path(
        sdf.metadata.get(
            "cache_path",
            str(sdf_cache_path.resolve() if sdf_cache_path is not None else default_sdf_cache_path(object_usd)),
        )
    ).resolve()
    object_query = make_object_query_from_sdf(sdf)
    hand_model = load_hand_kinematics_model(runtime_cfg)

    anchors = order_anchors_for_consumption(
        proposal_result.anchors,
        category=proposal_result.category,
    )[: max(1, int(top_k_anchors))]
    contact_results = resolve_contact_templates_for_anchors(
        runtime_cfg=runtime_cfg,
        anchors=anchors,
        samples=proposal_result.samples,
        descriptors=proposal_result.descriptors,
        contact_cfg_overrides=contact_cfg_overrides,
    )
    seed_results = generate_pose_seeds_for_contacts(
        runtime_cfg=runtime_cfg,
        contact_results=contact_results,
        num_seeds_per_contact=num_seeds_per_contact,
        seed=seed,
        semantic_hand_model=hand_model,
    )

    bundles: List[ContactSeedOptimizationBundle] = []
    requested_max_workers = 0 if max_workers is None else int(max_workers)
    candidate_seed_lists_all: List[List[Any]] = [
        list(seed_result.seeds[: max(1, int(top_k_seeds_per_contact))])
        for seed_result in seed_results
    ]
    if top_k_optimized_per_contact is None or int(top_k_optimized_per_contact) <= 0:
        optimization_limit = None
        candidate_seed_lists = candidate_seed_lists_all
    else:
        optimization_limit = max(1, int(top_k_optimized_per_contact))
        candidate_seed_lists = [
            seed_list[:optimization_limit]
            for seed_list in candidate_seed_lists_all
        ]

    num_candidates_total = int(sum(len(seed_list) for seed_list in candidate_seed_lists))
    effective_max_workers = _effective_candidate_workers(
        requested_max_workers=requested_max_workers,
        num_candidates_total=num_candidates_total,
    )
    requested_parallel_backend = str(parallel_backend).lower()
    effective_parallel_backend = "serial" if effective_max_workers <= 1 else requested_parallel_backend
    parallel_backend_fallback_reason = None
    effective_process_start_method = None

    if effective_max_workers <= 1:
        optimized_results_by_bundle: List[List[StagedGraspResult]] = [
            optimize_staged_grasps(
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
                seeds=candidate_seeds,
                object_query=object_query,
                hand_model=hand_model,
                max_workers=1,
            )
            for contact_result, candidate_seeds in zip(contact_results, candidate_seed_lists)
        ]
    else:
        indexed_results: List[List[Optional[StagedGraspResult]]] = [
            [None] * len(candidate_seeds)
            for candidate_seeds in candidate_seed_lists
        ]
        task_keys = [
            (bundle_idx, seed_idx)
            for bundle_idx, candidate_seeds in enumerate(candidate_seed_lists)
            for seed_idx, _seed in enumerate(candidate_seeds)
        ]
        if requested_parallel_backend == "process":
            try:
                start_methods = mp.get_all_start_methods()
                start_method = "fork" if "fork" in start_methods else "spawn"
                effective_process_start_method = start_method
                mp_context = mp.get_context(start_method)
                with ProcessPoolExecutor(
                    max_workers=effective_max_workers,
                    mp_context=mp_context,
                    initializer=_init_process_candidate_worker,
                    initargs=(
                        runtime_cfg,
                        contact_results,
                        candidate_seed_lists,
                        object_query,
                        hand_model,
                    ),
                ) as executor:
                    future_to_key = {
                        executor.submit(_compute_candidate_process, task_key): task_key
                        for task_key in task_keys
                    }
                    for future in as_completed(future_to_key):
                        bundle_idx, seed_idx, result = future.result()
                        indexed_results[bundle_idx][seed_idx] = result
            except Exception as exc:
                effective_parallel_backend = "thread"
                parallel_backend_fallback_reason = f"{type(exc).__name__}: {exc}"

        if effective_parallel_backend == "thread":
            with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
                future_to_key = {
                    executor.submit(
                        optimize_staged_grasp,
                        runtime_cfg=runtime_cfg,
                        contact_result=contact_results[bundle_idx],
                        seed=candidate_seed_lists[bundle_idx][seed_idx],
                        object_query=object_query,
                        hand_model=hand_model,
                    ): (bundle_idx, seed_idx)
                    for bundle_idx, seed_idx in task_keys
                }
                for future in as_completed(future_to_key):
                    bundle_idx, seed_idx = future_to_key[future]
                    indexed_results[bundle_idx][seed_idx] = future.result()

        optimized_results_by_bundle = []
        for contact_result, bundle_results in zip(contact_results, indexed_results):
            ordered_results = [result for result in bundle_results if result is not None]
            optimized_results_by_bundle.append(sort_staged_grasp_results(contact_result, ordered_results))

    for contact_result, seed_result, optimized_results in zip(
        contact_results,
        seed_results,
        optimized_results_by_bundle,
    ):
        bundles.append(
            ContactSeedOptimizationBundle(
                contact_result=contact_result,
                seed_result=seed_result,
                optimized_results=optimized_results,
            )
        )

    return SingleHandStagedPipelineResult(
        proposal_result=proposal_result,
        bundles=bundles,
        sdf_cache_path=effective_sdf_cache_path,
        metadata={
            "voxel_size": voxel_size,
            "padding_voxels": padding_voxels,
            "force_rebuild_sdf": force_rebuild_sdf,
            "top_k_anchors": int(top_k_anchors),
            "num_seeds_per_contact": None if num_seeds_per_contact is None else int(num_seeds_per_contact),
            "top_k_seeds_per_contact": int(top_k_seeds_per_contact),
            "top_k_optimized_per_contact": (
                None if optimization_limit is None else int(optimization_limit)
            ),
            "optimizer_cfg_overrides": deepcopy(optimizer_cfg_overrides) if optimizer_cfg_overrides else {},
            "num_generated_seeds_total": int(sum(len(seed_result.seeds) for seed_result in seed_results)),
            "num_optimized_pose_candidates_total": int(num_candidates_total),
            "parallelism": {
                "requested_max_workers": int(requested_max_workers),
                "parallel_scope": "global_candidates",
                "effective_max_workers_total": int(effective_max_workers),
                "requested_parallel_backend": requested_parallel_backend,
                "effective_parallel_backend": effective_parallel_backend,
                "effective_process_start_method": effective_process_start_method,
                "parallel_backend_fallback_reason": parallel_backend_fallback_reason,
            },
        },
    )


def run_single_hand_staged_cat1(
    *,
    hand_dir: Path,
    global_config_dir: Path,
    side: Side,
    object_usd: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
    num_surface_points: int = 100000,
    seed: int = 0,
    top_k_anchors: int = 5,
    num_seeds_per_contact: Optional[int] = None,
    top_k_seeds_per_contact: int = 3,
    top_k_optimized_per_contact: Optional[int] = None,
    max_workers: Optional[int] = None,
    sdf_cache_path: Optional[Path] = None,
    force_rebuild_sdf: bool = False,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
    optimizer_cfg_overrides: Optional[Dict[str, Any]] = None,
    parallel_backend: str = "thread",
) -> SingleHandStagedPipelineResult:
    return run_single_hand_staged(
        hand_dir=hand_dir,
        global_config_dir=global_config_dir,
        side=side,
        category="cat1",
        object_usd=object_usd,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        num_surface_points=num_surface_points,
        seed=seed,
        top_k_anchors=top_k_anchors,
        num_seeds_per_contact=num_seeds_per_contact,
        top_k_seeds_per_contact=top_k_seeds_per_contact,
        top_k_optimized_per_contact=top_k_optimized_per_contact,
        max_workers=max_workers,
        sdf_cache_path=sdf_cache_path,
        force_rebuild_sdf=force_rebuild_sdf,
        contact_cfg_overrides=contact_cfg_overrides,
        optimizer_cfg_overrides=optimizer_cfg_overrides,
        parallel_backend=parallel_backend,
    )


def summarize_single_hand_staged_pipeline(
    result: SingleHandStagedPipelineResult,
    *,
    top_k_optimized_per_contact: Optional[int] = 1,
) -> Dict[str, Any]:
    bundles_summary: List[Dict[str, Any]] = []
    for bundle in result.bundles:
        if top_k_optimized_per_contact is None or int(top_k_optimized_per_contact) <= 0:
            seed_top_k = len(bundle.seed_result.seeds)
            optimized_subset = list(bundle.optimized_results)
        else:
            limit = max(1, int(top_k_optimized_per_contact))
            seed_top_k = limit
            optimized_subset = bundle.optimized_results[:limit]
        bundles_summary.append(
            {
                "contact": summarize_contact_resolution(bundle.contact_result),
                "seed_generation": summarize_seed_generation(bundle.seed_result, top_k=seed_top_k),
                "num_optimized_results": int(len(bundle.optimized_results)),
                "num_successful_optimized_results": int(
                    sum(1 for opt_result in bundle.optimized_results if bool(opt_result.success))
                ),
                "optimized": [
                    summarize_staged_grasp_result(opt_result)
                    for opt_result in optimized_subset
                ],
            }
        )

    return {
        "side": result.proposal_result.side,
        "category": result.proposal_result.category,
        "object_usd": str(result.proposal_result.object_usd),
        "num_anchors_total": int(len(result.proposal_result.anchors)),
        "num_contact_bundles": int(len(result.bundles)),
        "sdf_cache_path": str(result.sdf_cache_path),
        "metadata": result.metadata,
        "bundles": bundles_summary,
    }
