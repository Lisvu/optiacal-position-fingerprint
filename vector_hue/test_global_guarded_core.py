#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks for global guarded selection and fallback."""

from __future__ import annotations

import numpy as np

import global_guarded_core as guarded


def make_candidate(candidate_id: int, route_raw_bers: list[float]) -> guarded.core.Candidate:
    raw = np.asarray(route_raw_bers, dtype=float)
    return guarded.core.Candidate(
        candidate_id=candidate_id,
        source="test",
        probes=np.asarray([5, 10, 15], dtype=float),
        authorized_position_bers=[0.0],
        route_keys=[(1, 2), (3, 2), (4, 2)],
        route_raw_bers=raw,
        route_min_bers=guarded.core.base.corrected_ber(raw),
    )


def make_candidate_with_routes(
    candidate_id: int,
    route_raw_bers: list[float],
    route_keys: list[tuple[int, int]],
) -> guarded.core.Candidate:
    raw = np.asarray(route_raw_bers, dtype=float)
    return guarded.core.Candidate(
        candidate_id=candidate_id,
        source="test",
        probes=np.asarray([5, 10, 15], dtype=float),
        authorized_position_bers=[0.0],
        route_keys=route_keys,
        route_raw_bers=raw,
        route_min_bers=guarded.core.base.corrected_ber(raw),
    )


def test_anchor_worst_routes() -> None:
    anchor = make_candidate(1, [0.34, 0.41, 0.39])
    routes = guarded.anchor_worst_routes(anchor, 2)
    assert routes == [(1, 2), (4, 2)]


def test_adaptive_weak_route_count_scales_with_route_count() -> None:
    assert guarded.adaptive_weak_route_count(32, 5) == 5
    assert guarded.adaptive_weak_route_count(80, 5) == 12


def test_group_rescuer_beats_single_route_polluter() -> None:
    routes = [(idx, 2) for idx in range(8)]
    anchor = make_candidate_with_routes(1, [0.34, 0.35, 0.36, 0.37, 0.45, 0.45, 0.45, 0.45], routes)
    single_route_polluter = make_candidate_with_routes(2, [0.49, 0.02, 0.02, 0.02, 0.45, 0.45, 0.45, 0.45], routes)
    group_rescuer = make_candidate_with_routes(3, [0.33, 0.33, 0.33, 0.33, 0.44, 0.44, 0.44, 0.44], routes)

    weak_routes = guarded.anchor_worst_routes(anchor, 4)
    assert guarded.group_rescue_score(group_rescuer, weak_routes) > guarded.group_rescue_score(single_route_polluter, weak_routes)

    selected, _ = guarded.select_global_guarded_candidates(
        [anchor, single_route_polluter, group_rescuer],
        selected_count=2,
        weak_routes=weak_routes,
    )
    assert [candidate.candidate_id for candidate in selected] == [1, 3]


def test_hybrid_selection_keeps_bounded_route_rescuers_for_mixing() -> None:
    routes = [(idx, 2) for idx in range(6)]
    anchor = make_candidate_with_routes(1, [0.30, 0.31, 0.32, 0.33, 0.44, 0.44], routes)
    top_global = make_candidate_with_routes(2, [0.29, 0.32, 0.33, 0.34, 0.44, 0.44], routes)
    bounded_rescuer = make_candidate_with_routes(3, [0.47, 0.20, 0.21, 0.22, 0.43, 0.43], routes)
    extreme_polluter = make_candidate_with_routes(4, [0.49, 0.02, 0.02, 0.02, 0.43, 0.43], routes)

    selected, _ = guarded.select_global_guarded_candidates(
        [anchor, top_global, bounded_rescuer, extreme_polluter],
        selected_count=3,
        weak_routes=guarded.anchor_worst_routes(anchor, 4),
    )
    selected_ids = [candidate.candidate_id for candidate in selected]
    assert 1 in selected_ids
    assert 3 in selected_ids
    assert 4 not in selected_ids


def test_hybrid_selection_keeps_specific_route_rescuer() -> None:
    routes = [(idx, 2) for idx in range(6)]
    anchor = make_candidate_with_routes(1, [0.30, 0.31, 0.32, 0.33, 0.44, 0.44], routes)
    top_global = make_candidate_with_routes(2, [0.29, 0.32, 0.33, 0.34, 0.44, 0.44], routes)
    group_rescuer_1 = make_candidate_with_routes(3, [0.29, 0.29, 0.29, 0.29, 0.43, 0.43], routes)
    group_rescuer_2 = make_candidate_with_routes(4, [0.28, 0.28, 0.28, 0.28, 0.43, 0.43], routes)
    specific_route_rescuer = make_candidate_with_routes(5, [0.49, 0.12, 0.12, 0.12, 0.43, 0.43], routes)

    selected, _ = guarded.select_global_guarded_candidates(
        [anchor, top_global, group_rescuer_1, group_rescuer_2, specific_route_rescuer],
        selected_count=4,
        weak_routes=guarded.anchor_worst_routes(anchor, 4),
    )
    assert 5 in [candidate.candidate_id for candidate in selected]


def test_selection_preserves_anchor() -> None:
    weak_star = make_candidate(1, [0.49, 0.12, 0.46])
    anchor = make_candidate(2, [0.34, 0.35, 0.36])
    route_rescuer = make_candidate(3, [0.20, 0.48, 0.20])
    selected, selected_anchor = guarded.select_global_guarded_candidates(
        [weak_star, anchor, route_rescuer],
        selected_count=2,
        weak_routes=[(1, 2)],
    )
    assert selected_anchor.candidate_id == anchor.candidate_id
    assert anchor.candidate_id in [candidate.candidate_id for candidate in selected]


def test_anchor_floor_fallback() -> None:
    raw_matrix = np.asarray([
        [0.30, 0.32],
        [0.31, 0.33],
    ], dtype=float)
    ratio, _, mixed_min, optimizer, floor_applied, diagnostic = guarded.optimize_ratio_with_diagnostic(
        raw_matrix,
        anchor_index=1,
        anchor_floor=0.40,
    )

    assert floor_applied is True
    assert optimizer == "anchor_only_floor"
    assert diagnostic.startswith("floor_applied")
    assert np.isclose(ratio[1], 1.0)
    assert np.isclose(float(np.min(mixed_min)), 0.32)


def test_optimizer_diagnostic_reports_invalid_matrix() -> None:
    raw_matrix = np.asarray([[0.3, np.nan], [0.4, 0.2]], dtype=float)
    ratio, mixed_raw, mixed_min, optimizer, floor_applied, diagnostic = guarded.optimize_ratio_with_diagnostic(
        raw_matrix,
        anchor_index=0,
        anchor_floor=0.3,
    )
    assert optimizer == "anchor_only_floor"
    assert floor_applied is True
    assert diagnostic.startswith("invalid_raw_matrix")
    assert np.isclose(ratio[0], 1.0)


def test_local_optimizer_finds_mixed_solution() -> None:
    raw_matrix = np.asarray([
        [0.25, 0.50],
        [0.50, 0.25],
    ], dtype=float)
    ratio, mixed_raw, mixed_min, optimizer, floor_applied, diagnostic = guarded.optimize_ratio_with_diagnostic(
        raw_matrix,
        anchor_index=0,
        anchor_floor=0.25,
    )
    assert optimizer == "linprog"
    assert floor_applied is False
    assert diagnostic == "ok"
    assert np.allclose(ratio, [0.5, 0.5])
    assert np.allclose(mixed_raw, [0.375, 0.375])
    assert float(np.min(mixed_min)) > 0.25


def test_optimizer_does_not_depend_on_core_fallback() -> None:
    raw_matrix = np.asarray([
        [0.25, 0.50],
        [0.50, 0.25],
    ], dtype=float)
    original_optimizer = guarded.core.base.optimize_usage_ratio

    def bad_core_optimizer(matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
        return np.asarray([1.0, 0.0], dtype=float), 0.25, "best_single_fallback"

    guarded.core.base.optimize_usage_ratio = bad_core_optimizer
    try:
        ratio, _, mixed_min, optimizer, floor_applied, diagnostic = guarded.optimize_ratio_with_diagnostic(
            raw_matrix,
            anchor_index=0,
            anchor_floor=0.25,
        )
    finally:
        guarded.core.base.optimize_usage_ratio = original_optimizer

    assert optimizer == "linprog"
    assert floor_applied is False
    assert diagnostic == "ok"
    assert np.allclose(ratio, [0.5, 0.5])
    assert float(np.min(mixed_min)) > 0.25


def test_args_from_namespace_accepts_dataset_dir() -> None:
    class Namespace:
        sample_size = 3
        target_effective_k = 8
        baseline_probe_subset_count = 20
        targeted_probe_subset_count = 50
        base_mapping_count = 20
        baseline_mappings_per_subset = 8
        targeted_mappings_per_subset = 16
        selected_count = 20
        weak_route_count = 5
        eval_bits = 1000
        overwrite = False
        dataset_dir = r"data\mate40pro\high"

    args = guarded.args_from_namespace(7, "out", Namespace())
    assert args.dataset_dir == r"data\mate40pro\high"


if __name__ == "__main__":
    test_anchor_worst_routes()
    test_adaptive_weak_route_count_scales_with_route_count()
    test_group_rescuer_beats_single_route_polluter()
    test_hybrid_selection_keeps_bounded_route_rescuers_for_mixing()
    test_hybrid_selection_keeps_specific_route_rescuer()
    test_selection_preserves_anchor()
    test_anchor_floor_fallback()
    test_optimizer_diagnostic_reports_invalid_matrix()
    test_local_optimizer_finds_mixed_solution()
    test_optimizer_does_not_depend_on_core_fallback()
    test_args_from_namespace_accepts_dataset_dir()
    print("global guarded core checks passed")
