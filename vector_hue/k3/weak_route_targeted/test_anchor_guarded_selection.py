#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks for anchor-preserved weak-route selection."""

from __future__ import annotations

import numpy as np

import run_weak_route_targeted_k3 as targeted


def make_candidate(candidate_id: int, route_raw_bers: list[float]) -> targeted.core.Candidate:
    raw = np.asarray(route_raw_bers, dtype=float)
    return targeted.core.Candidate(
        candidate_id=candidate_id,
        source="test",
        probes=np.asarray([5, 10, 15], dtype=float),
        authorized_position_bers=[0.0],
        route_keys=[(1, 7), (2, 7), (3, 7)],
        route_raw_bers=raw,
        route_min_bers=targeted.core.base.corrected_ber(raw),
    )


def test_selection_keeps_global_anchor() -> None:
    weak_route_star = make_candidate(1, [0.49, 0.12, 0.46])
    anchor = make_candidate(2, [0.34, 0.35, 0.36])
    route_rescuer = make_candidate(3, [0.20, 0.48, 0.20])

    selected, selected_anchor = targeted.select_global_guarded_candidates(
        [weak_route_star, anchor, route_rescuer],
        selected_count=2,
        weak_routes=[(1, 7)],
    )

    assert selected_anchor.candidate_id == anchor.candidate_id
    assert selected[0].candidate_id == anchor.candidate_id
    assert anchor.candidate_id in [candidate.candidate_id for candidate in selected]


def test_anchor_floor_fallback() -> None:
    weak_route_star = make_candidate(1, [0.49, 0.12, 0.46])
    anchor = make_candidate(2, [0.34, 0.35, 0.36])
    selected = [weak_route_star, anchor]

    original_optimizer = targeted.core.base.optimize_usage_ratio

    def bad_optimizer(raw_matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
        return np.asarray([1.0, 0.0], dtype=float), 0.12, "forced_bad_optimizer"

    targeted.core.base.optimize_usage_ratio = bad_optimizer
    try:
        ratio, _, mixed_min, optimizer, floor_applied = targeted.anchor_guarded_ratio_and_min(selected, anchor)
    finally:
        targeted.core.base.optimize_usage_ratio = original_optimizer

    assert floor_applied is True
    assert optimizer == "anchor_only_floor"
    assert np.isclose(ratio[1], 1.0)
    assert float(np.min(mixed_min)) >= anchor.min_route_min_ber


if __name__ == "__main__":
    test_selection_keeps_global_anchor()
    test_anchor_floor_fallback()
    print("anchor guarded selection checks passed")
