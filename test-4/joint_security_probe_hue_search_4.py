#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint security-aware probe and hue-mapping search for 4 legal positions.

Redesigned for reliability + speed:
  1. All raw matrices pre-loaded into memory.
  2. Fingerprint models cached via LRU to avoid repeated SVD.
  3. Proxy evaluation uses 512 random non-FEC blocks (statistically reliable,
     ~10x faster than FEC with Viterbi).
  4. Illegal evaluation iterates ALL illegal positions in proxy stage.
  5. Probe search:
       - random-sampling with proxy-screening (no expensive inner loop)
       - greedy hue-mapping local search on proxy
       - light SA (30 iters) on top-3 candidates only
       - 1-round local hill-climb
  6. Two confirmation tiers:
       - Mid: 2000 bits FEC
       - Final: 10000 bits FEC (only for mid-pass candidates)

Target:
  legal BER < 0.001
  min_illegal_ber > 0.1
"""

from __future__ import annotations

import csv
import functools
import itertools
import os
import random
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import test_4_simple as test


LEGAL_POSITION_COUNT = 4
LIGHT_CONDITION = "white"
OUTPUT_RESULTS_FILENAME = "joint_security_probe_hue_results_4.csv"
SAFE_PROBE_RESULTS_FILENAME = "joint_security_safe_probes_4.csv"

TARGET_LEGAL_BER = 0.001
MIN_ILLEGAL_BER = 0.1

# ---------------------------------------------------------------------------
# Evaluation budgets
# ---------------------------------------------------------------------------
PROXY_NUM_BLOCKS = 512
MID_LEGAL_BITS = 2000
MID_ILLEGAL_BITS = 2000
FINAL_LEGAL_BITS = 10000
FINAL_ILLEGAL_BITS = 10000

# ---------------------------------------------------------------------------
# Search hyper-parameters
# ---------------------------------------------------------------------------
PROBE_COUNTS = [8, 9, 10, 11, 12, 13, 7, 14, 6, 15]
RANDOM_SAMPLES_PER_COUNT = 30
SA_ITERATIONS = 30
SA_START_TEMP = 0.06
SA_COOLING = 0.9
LOCAL_ROUNDS = 1
LOCAL_NEIGHBORS = 8

SELECTION_SEED = 20260507

# ---------------------------------------------------------------------------
# Pre-loaded data cache
# ---------------------------------------------------------------------------
_RAW_MATRICES: dict[int, np.ndarray] = {}
_ALL_POSITIONS: list[int] = []


def _preload_matrices(project_root: str) -> None:
    global _RAW_MATRICES, _ALL_POSITIONS
    if _RAW_MATRICES:
        return
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        if entry.endswith(".csv") and os.path.splitext(entry)[0].isdigit():
            positions.append(int(os.path.splitext(entry)[0]))
    _ALL_POSITIONS = sorted(positions)
    for pos in _ALL_POSITIONS:
        csv_path = os.path.join(data_dir, f"{pos}.csv")
        _RAW_MATRICES[pos] = pd.read_csv(csv_path).values.astype(float)


# ---------------------------------------------------------------------------
# Cached model builders
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=512)
def _cached_build_models(positions_key: tuple[int, ...], probes_key: tuple[float, ...]) -> list[test.FingerprintModel]:
    probes_array = np.asarray(probes_key, dtype=float)
    row_indices = [int((p / 5) - 1) for p in probes_key]
    models: list[test.FingerprintModel] = []
    for pos in positions_key:
        mat = _RAW_MATRICES[pos][row_indices].astype(float, copy=False)
        models.append(test.extract_fingerprint(probes_array, mat, force_positive_first=True))
    return test.align_model_directions(models)


def build_legal_models_fast(legal_positions: Sequence[int], probes: Sequence[float]) -> list[test.FingerprintModel]:
    return _cached_build_models(tuple(legal_positions), tuple(sorted(probes)))


def build_illegal_models_fast(legal_positions: Sequence[int], probes: Sequence[float]) -> dict[int, test.FingerprintModel]:
    pkey = tuple(sorted(probes))
    lkey = set(legal_positions)
    illegal_models: dict[int, test.FingerprintModel] = {}
    for pos in _ALL_POSITIONS:
        if pos in lkey:
            continue
        illegal_models[pos] = _cached_build_models((pos,), pkey)[0]
    return illegal_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_available_positions() -> list[int]:
    return list(_ALL_POSITIONS)


def get_all_probes(legal_positions: Sequence[int]) -> list[float]:
    n_rows = _RAW_MATRICES[legal_positions[0]].shape[0]
    return (5 + np.arange(n_rows) * 5).astype(float).tolist()


def min_interval_for_probe_count(probe_count: int) -> int:
    if probe_count <= 12:
        return 30
    if probe_count <= 16:
        return 20
    if probe_count <= 22:
        return 15
    return 10


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict[tuple[int, ...], int] | None) -> str:
    if not hue_mapping:
        return ""
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


def score_proxy(legal_ber: float, min_illegal_ber: float | None) -> float:
    if legal_ber <= 0.005 and min_illegal_ber is not None and min_illegal_ber > MIN_ILLEGAL_BER:
        return 100.0 + min_illegal_ber * 100.0
    legal_penalty = max(0.0, legal_ber - 0.005) * 500.0
    illegal_penalty = 0.0
    if min_illegal_ber is not None and min_illegal_ber <= MIN_ILLEGAL_BER:
        illegal_penalty = (MIN_ILLEGAL_BER - min_illegal_ber) * 300.0
    return -legal_penalty - illegal_penalty


# ---------------------------------------------------------------------------
# Random bit-block cache (re-used to avoid re-generation)
# ---------------------------------------------------------------------------
_PROXY_BLOCKS: dict[tuple[int, int, int], list[np.ndarray]] = {}


def _get_proxy_blocks(num_blocks: int, num_positions: int, seed: int) -> list[np.ndarray]:
    key = (num_blocks, num_positions, seed)
    if key not in _PROXY_BLOCKS:
        rng = random.Random(seed)
        _PROXY_BLOCKS[key] = [
            np.asarray([rng.choice([-1, 1]) for _ in range(num_positions)], dtype=int)
            for _ in range(num_blocks)
        ]
    return _PROXY_BLOCKS[key]


# ---------------------------------------------------------------------------
# Proxy evaluation (512 random non-FEC blocks, all illegal positions)
# ---------------------------------------------------------------------------
def proxy_evaluate_legal(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    seed: int = 1,
) -> float:
    return float(test.evaluate_blocks_ber(
        list(legal_models),
        _get_proxy_blocks(PROXY_NUM_BLOCKS, len(legal_models), seed),
        hue_mapping,
    ))


def proxy_evaluate_illegal(
    legal_models: Sequence[test.FingerprintModel],
    illegal_models: dict[int, test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    probes: Sequence[float],
    seed: int = 1,
) -> dict:
    probes_array = np.asarray(probes, dtype=float)
    legal_codes = [model.code for model in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    bit_blocks_pm = _get_proxy_blocks(PROXY_NUM_BLOCKS, len(legal_models), seed)

    global_min = float("inf")
    raw_at_min = 0.0
    worst_pos = None
    worst_idx = None
    worst_vec: list[float] | None = None
    all_secure: list[float] = []

    for pos, model in illegal_models.items():
        errors = np.zeros(len(legal_models), dtype=float)
        totals = np.zeros(len(legal_models), dtype=float)
        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            # 非法位置也尝试用多主成分解码（模拟最强攻击者）
            if model.W is not None and model.W.ndim > 1:
                dec = test.decode_local_block(obs, model.W, model.multi_code)
            else:
                dec = test.decode_local_block(obs, model.w, model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for idx in range(len(legal_models)):
                totals[idx] += 1
                if dec.bit_hat_bin != true_bits[idx]:
                    errors[idx] += 1
        raw_bers = errors / np.maximum(totals, 1.0)
        secure_bers = np.minimum(raw_bers, 1.0 - raw_bers)
        all_secure.extend(secure_bers.tolist())
        local_min = float(np.min(secure_bers))
        local_idx = int(np.argmin(secure_bers))
        if local_min < global_min:
            global_min = local_min
            raw_at_min = float(raw_bers[local_idx])
            worst_pos = pos
            worst_idx = local_idx
            worst_vec = [float(v) for v in raw_bers]

    return {
        "min_illegal_ber": float(global_min if all_secure else 0.0),
        "raw_ber_at_min": float(raw_at_min),
        "average_illegal_ber": float(np.mean(all_secure)) if all_secure else 0.0,
        "worst_illegal_position": worst_pos,
        "worst_legal_index": worst_idx,
        "worst_illegal_ber_vector": worst_vec or [],
    }


def proxy_evaluate_probe_set(
    legal_positions: Sequence[int],
    probes: Sequence[float],
    seed: int = 1,
) -> dict:
    legal_models = build_legal_models_fast(legal_positions, probes)
    illegal_models = build_illegal_models_fast(legal_positions, probes)

    # Build hue mapping via test.build_hue_mapping (fast init) then 2-round greedy local
    probes_array = np.asarray(probes, dtype=float)
    hue_mapping_init = test.build_hue_mapping(
        list(legal_models), probes_array, mapping_eval_bits=200, top_k_per_combination=2, rng=random.Random(seed)
    )
    hue_mapping = _greedy_local_search_mapping(legal_models, probes, hue_mapping_init, seed)

    legal_ber = proxy_evaluate_legal(legal_models, hue_mapping, seed)

    if legal_ber > 0.02:  # quick reject
        return {
            "hue_mapping": hue_mapping,
            "legal_ber": legal_ber,
            "min_illegal_ber": None,
            "score": score_proxy(legal_ber, None),
            "security_satisfied": False,
        }

    security = proxy_evaluate_illegal(legal_models, illegal_models, hue_mapping, probes, seed)
    return {
        "hue_mapping": hue_mapping,
        "legal_ber": legal_ber,
        "score": score_proxy(legal_ber, security["min_illegal_ber"]),
        "security_satisfied": legal_ber <= 0.005 and security["min_illegal_ber"] > MIN_ILLEGAL_BER,
        **security,
    }


def _generate_relaxed_candidates(
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
    top_k: int = 3,
) -> dict[tuple[int, ...], list[int]]:
    probes_array = np.asarray(probes, dtype=float)
    # 使用有效z值（多主成分混合后的投影，与发送端 eff_code 对应）
    z_list = []
    for model in legal_models:
        if model.Z is not None and model.Z.ndim > 1:
            k = model.Z.shape[1]
            alpha = test.ALPHA[:k] if hasattr(test, 'ALPHA') else np.ones(k)
            alpha = np.asarray(alpha, dtype=float)
            z_list.append(np.asarray(model.Z @ alpha, dtype=float))
        else:
            z_list.append(np.asarray(model.z, dtype=float))
    combinations = list(itertools.product([1, -1], repeat=len(legal_models)))
    strict = test.generate_mapping_candidates(list(legal_models), probes_array, top_k_per_combination=2)
    cmap: dict[tuple[int, ...], list[int]] = {}
    for comb in combinations:
        scored = []
        for pi, probe in enumerate(probes_array):
            margins = [int(sign) * float(z[pi]) for sign, z in zip(comb, z_list)]
            matched = sum(1 for v in margins if v > 0)
            margin_sum = sum(abs(v) for v in margins)
            penalty = sum(abs(v) for v in margins if v <= 0)
            scored.append((matched * 1000.0 + margin_sum - penalty * 2.0, int(probe)))
        scored.sort(reverse=True)
        cands: list[int] = []
        for probe in strict.get(comb, []):
            if probe not in cands:
                cands.append(int(probe))
        for _, probe in scored:
            if probe not in cands:
                cands.append(int(probe))
            if len(cands) >= top_k:
                break
        cmap[comb] = cands
    return cmap


def _greedy_local_search_mapping(
    legal_models: Sequence[test.FingerprintModel],
    probes: Sequence[float],
    initial: dict[tuple[int, ...], int],
    seed: int,
    rounds: int = 2,
) -> dict[tuple[int, ...], int]:
    cmap = _generate_relaxed_candidates(legal_models, probes, top_k=3)
    current = dict(initial)
    current_ber = proxy_evaluate_legal(legal_models, current, seed)
    keys = sorted(cmap.keys())
    for _ in range(rounds):
        improved = False
        for key in keys:
            best_probe = current[key]
            best_ber = current_ber
            for cand in cmap[key]:
                if cand == best_probe:
                    continue
                trial = dict(current)
                trial[key] = int(cand)
                ber = proxy_evaluate_legal(legal_models, trial, seed)
                if ber < best_ber:
                    best_ber = ber
                    best_probe = int(cand)
                    improved = True
            current[key] = best_probe
            current_ber = best_ber
        if not improved:
            break
    return current


# ---------------------------------------------------------------------------
# FEC evaluation (mid / final)
# ---------------------------------------------------------------------------
def fec_evaluate_legal(
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    num_bits: int,
    rng: random.Random,
) -> float:
    info_bits = test.generate_random_information_bits(num_bits, len(legal_models), rng=rng)
    return float(test.evaluate_blocks_ber_with_convolutional_fec(list(legal_models), info_bits, hue_mapping))


def fec_evaluate_illegal(
    legal_models: Sequence[test.FingerprintModel],
    illegal_models: dict[int, test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    probes: Sequence[float],
    num_bits: int,
    rng: random.Random,
) -> dict:
    probes_array = np.asarray(probes, dtype=float)
    legal_codes = [model.code for model in legal_models]
    probe_to_row = test.build_probe_to_row(probes_array)
    info_bits = test.generate_random_information_bits(num_bits, len(legal_models), rng=rng)
    bit_blocks_pm = test.build_convolutional_bit_blocks(info_bits)

    global_min = float("inf")
    raw_at_min = 0.0
    worst_pos = None
    worst_idx = None
    worst_vec: list[float] | None = None
    all_secure: list[float] = []

    for pos, model in illegal_models.items():
        received: list[int] = []
        for bits_pm in bit_blocks_pm:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, model.Y, probe_to_row)
            # 非法位置也尝试用多主成分解码（模拟最强攻击者）
            if model.W is not None and model.W.ndim > 1:
                dec = test.decode_local_block(obs, model.W, model.multi_code)
            else:
                dec = test.decode_local_block(obs, model.w, model.code)
            received.append(int(dec.bit_hat_bin))
        decoded = test.viterbi_decode_hard(received)
        raw_bers = []
        for li in range(len(legal_models)):
            ref = info_bits[:, li]
            cl = min(len(decoded), len(ref))
            if cl <= 0:
                raw_bers.append(0.0)
                continue
            raw_bers.append(float(np.mean(decoded[:cl] != ref[:cl])))
        secure = [min(ber, 1.0 - ber) for ber in raw_bers]
        all_secure.extend(secure)
        local_min = float(min(secure))
        local_idx = int(np.argmin(secure))
        if local_min < global_min:
            global_min = local_min
            raw_at_min = float(raw_bers[local_idx])
            worst_pos = pos
            worst_idx = local_idx
            worst_vec = [float(v) for v in raw_bers]

    return {
        "min_illegal_ber": float(global_min if all_secure else 0.0),
        "raw_ber_at_min": float(raw_at_min),
        "average_illegal_ber": float(np.mean(all_secure)) if all_secure else 0.0,
        "worst_illegal_position": worst_pos,
        "worst_legal_index": worst_idx,
        "worst_illegal_ber_vector": worst_vec or [],
    }


def fec_evaluate_probe_set(
    legal_positions: Sequence[int],
    probes: Sequence[float],
    hue_mapping: dict[tuple[int, ...], int],
    legal_bits: int,
    illegal_bits: int,
    rng: random.Random,
) -> dict:
    legal_models = build_legal_models_fast(legal_positions, probes)
    illegal_models = build_illegal_models_fast(legal_positions, probes)
    legal_ber = fec_evaluate_legal(legal_models, hue_mapping, legal_bits, rng)
    if legal_ber > TARGET_LEGAL_BER * 2:
        return {
            "hue_mapping": hue_mapping,
            "legal_ber": legal_ber,
            "min_illegal_ber": None,
            "score": score_proxy(legal_ber, None),
            "security_satisfied": False,
        }
    security = fec_evaluate_illegal(legal_models, illegal_models, hue_mapping, probes, illegal_bits, rng)
    return {
        "hue_mapping": hue_mapping,
        "legal_ber": legal_ber,
        "score": score_proxy(legal_ber, security["min_illegal_ber"]),
        "security_satisfied": legal_ber <= TARGET_LEGAL_BER and security["min_illegal_ber"] > MIN_ILLEGAL_BER,
        **security,
    }


# ---------------------------------------------------------------------------
# Probe search
# ---------------------------------------------------------------------------
def random_valid_probe_set(all_probes: Sequence[float], probe_count: int, rng: random.Random) -> np.ndarray:
    min_interval = min_interval_for_probe_count(probe_count)
    for _ in range(50):
        probes = np.sort(np.asarray(rng.sample(all_probes, probe_count), dtype=float))
        if test.is_valid_probe_set(probes, min_interval=min_interval):
            return probes
    # last resort
    return np.sort(np.asarray(rng.sample(all_probes, probe_count), dtype=float))


def simulated_annealing_probe_search(
    legal_positions: Sequence[int],
    initial: Sequence[float],
    all_probes: Sequence[float],
    rng: random.Random,
    iterations: int = SA_ITERATIONS,
    start_temp: float = SA_START_TEMP,
    cooling: float = SA_COOLING,
) -> tuple[np.ndarray, dict]:
    current = np.sort(np.asarray(initial, dtype=float))
    current_eval = proxy_evaluate_probe_set(legal_positions, current, seed=rng.randint(0, 2**31))
    best = current.copy()
    best_eval = dict(current_eval)

    temp = start_temp
    for _ in range(iterations):
        if best_eval.get("security_satisfied"):
            break
        idx = rng.randrange(len(current))
        pool = [p for p in all_probes if p not in current]
        if not pool:
            break
        cand = float(rng.choice(pool))
        trial = current.copy()
        trial[idx] = cand
        trial = np.sort(trial)
        if not test.is_valid_probe_set(trial, min_interval=min_interval_for_probe_count(len(trial))):
            temp = max(temp * cooling, 1e-5)
            continue
        trial_eval = proxy_evaluate_probe_set(legal_positions, trial, seed=rng.randint(0, 2**31))
        delta = trial_eval["score"] - current_eval["score"]
        accept = delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-6))
        if accept:
            current = trial
            current_eval = trial_eval
            if current_eval["score"] > best_eval["score"]:
                best = current.copy()
                best_eval = dict(current_eval)
        temp = max(temp * cooling, 1e-5)
    return best, best_eval


def local_probe_refinement(
    legal_positions: Sequence[int],
    initial: Sequence[float],
    all_probes: Sequence[float],
    rng: random.Random,
    rounds: int = LOCAL_ROUNDS,
    neighbors: int = LOCAL_NEIGHBORS,
) -> tuple[np.ndarray, dict]:
    current = np.sort(np.asarray(initial, dtype=float))
    current_eval = proxy_evaluate_probe_set(legal_positions, current, seed=rng.randint(0, 2**31))
    for _ in range(rounds):
        improved = False
        for idx in range(len(current)):
            pool = [p for p in all_probes if p not in current]
            if not pool:
                continue
            for cand in rng.sample(pool, min(neighbors, len(pool))):
                trial = current.copy()
                trial[idx] = cand
                trial = np.sort(trial)
                if not test.is_valid_probe_set(trial, min_interval=min_interval_for_probe_count(len(trial))):
                    continue
                ev = proxy_evaluate_probe_set(legal_positions, trial, seed=rng.randint(0, 2**31))
                if ev["score"] > current_eval["score"]:
                    current = trial
                    current_eval = ev
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current, current_eval


def search_security_aware_probes(
    project_root: str,
    legal_positions: Sequence[int],
    rng: random.Random,
) -> dict:
    _preload_matrices(project_root)
    all_probes = get_all_probes(legal_positions)
    best_candidate: dict | None = None
    shortlisted: list[dict] = []

    for probe_count in PROBE_COUNTS:
        if probe_count > len(all_probes):
            continue
        print(f"    probe_count={probe_count}")
        epoch_best: dict | None = None

        # --- Random sampling with proxy screening ---
        for sample_idx in range(RANDOM_SAMPLES_PER_COUNT):
            probes = random_valid_probe_set(all_probes, probe_count, rng)
            ev = proxy_evaluate_probe_set(legal_positions, probes, seed=rng.randint(0, 2**31))
            rec = {
                "probes": probes.copy(),
                "probe_count": probe_count,
                "min_interval": min_interval_for_probe_count(probe_count),
                **ev,
            }
            if best_candidate is None or rec["score"] > best_candidate["score"]:
                best_candidate = rec
            if epoch_best is None or rec["score"] > epoch_best["score"]:
                epoch_best = rec
            if rec.get("security_satisfied"):
                shortlisted.append(rec)

        if epoch_best is None:
            continue

        # --- SA on the best of this probe_count ---
        sa_probes, sa_eval = simulated_annealing_probe_search(
            legal_positions, epoch_best["probes"], all_probes, rng
        )
        sa_rec = {
            "probes": sa_probes.copy(),
            "probe_count": probe_count,
            "min_interval": min_interval_for_probe_count(probe_count),
            **sa_eval,
        }
        if sa_rec["score"] > best_candidate["score"]:
            best_candidate = sa_rec
        if sa_rec.get("security_satisfied"):
            shortlisted.append(sa_rec)

    # --- Local refinement on top candidates ---
    candidates_to_refine = [best_candidate] + sorted(
        shortlisted, key=lambda x: x["score"], reverse=True
    )[:4]
    refined: list[dict] = []
    for cand in candidates_to_refine:
        if cand is None:
            continue
        ref_probes, ref_eval = local_probe_refinement(
            legal_positions, cand["probes"], all_probes, rng
        )
        ref_rec = {
            "probes": ref_probes.copy(),
            "probe_count": len(ref_probes),
            "min_interval": min_interval_for_probe_count(len(ref_probes)),
            **ref_eval,
        }
        refined.append(ref_rec)
        if ref_rec["score"] > best_candidate["score"]:
            best_candidate = ref_rec

    # --- Mid FEC confirmation (2000 bits) on top 4 ---
    confirm_pool = sorted([best_candidate] + refined, key=lambda x: x["score"], reverse=True)[:4]
    for cand in confirm_pool:
        if cand is None:
            continue
        mid_eval = fec_evaluate_probe_set(
            legal_positions, cand["probes"], cand["hue_mapping"],
            legal_bits=MID_LEGAL_BITS, illegal_bits=MID_ILLEGAL_BITS, rng=rng,
        )
        mid_rec = {
            "probes": cand["probes"].copy(),
            "probe_count": len(cand["probes"]),
            "min_interval": min_interval_for_probe_count(len(cand["probes"])),
            **mid_eval,
        }
        if mid_rec["score"] > (best_candidate["score"] if best_candidate else float("-inf")):
            best_candidate = mid_rec
        if mid_rec.get("security_satisfied"):
            # Final confirmation
            final_eval = fec_evaluate_probe_set(
                legal_positions, mid_rec["probes"], mid_rec["hue_mapping"],
                legal_bits=FINAL_LEGAL_BITS, illegal_bits=FINAL_ILLEGAL_BITS, rng=rng,
            )
            final_rec = {
                "probes": mid_rec["probes"].copy(),
                "probe_count": len(mid_rec["probes"]),
                "min_interval": min_interval_for_probe_count(len(mid_rec["probes"])),
                **final_eval,
            }
            if final_rec.get("security_satisfied"):
                return final_rec

    if best_candidate is None:
        raise RuntimeError("No probe candidates were evaluated")

    # Fallback final confirmation on absolute best
    final_eval = fec_evaluate_probe_set(
        legal_positions, best_candidate["probes"], best_candidate["hue_mapping"],
        legal_bits=FINAL_LEGAL_BITS, illegal_bits=FINAL_ILLEGAL_BITS, rng=rng,
    )
    return {
        "probes": best_candidate["probes"].copy(),
        "probe_count": len(best_candidate["probes"]),
        "min_interval": min_interval_for_probe_count(len(best_candidate["probes"])),
        **final_eval,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def generate_position_combinations(project_root: str) -> list[tuple[int, ...]]:
    _preload_matrices(project_root)
    return list(itertools.combinations(get_available_positions(), LEGAL_POSITION_COUNT))


def load_existing_results(results_file: str) -> list[dict]:
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination", "probe_count", "min_interval", "probes", "hue_mapping",
        "legal_ber", "min_illegal_ber", "raw_ber_at_min", "average_illegal_ber",
        "worst_illegal_position", "worst_legal_position", "worst_illegal_ber_vector",
        "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_safe_probe_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "position_combination", "safe_probe_count", "safe_min_interval", "safe_probes", "safe_hue_mapping",
        "legal_ber", "min_illegal_ber", "raw_ber_at_min", "average_illegal_ber",
        "worst_illegal_position", "worst_legal_position", "worst_illegal_ber_vector",
    ]
    safe_rows = []
    for row in rows:
        if row.get("security_satisfied") != "yes":
            continue
        safe_rows.append({
            "position_combination": row["position_combination"],
            "safe_probe_count": row["probe_count"],
            "safe_min_interval": row["min_interval"],
            "safe_probes": row["probes"],
            "safe_hue_mapping": row["hue_mapping"],
            "legal_ber": row["legal_ber"],
            "min_illegal_ber": row["min_illegal_ber"],
            "raw_ber_at_min": row["raw_ber_at_min"],
            "average_illegal_ber": row["average_illegal_ber"],
            "worst_illegal_position": row["worst_illegal_position"],
            "worst_legal_position": row["worst_legal_position"],
            "worst_illegal_ber_vector": row["worst_illegal_ber_vector"],
        })
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safe_rows)


def row_from_result(legal_positions: Sequence[int], result: dict) -> dict:
    worst_legal_index = result.get("worst_legal_index")
    worst_legal_position = "" if worst_legal_index is None else legal_positions[int(worst_legal_index)]
    return {
        "position_combination": str(tuple(legal_positions)),
        "probe_count": result["probe_count"],
        "min_interval": result["min_interval"],
        "probes": format_probes(result["probes"]),
        "hue_mapping": format_hue_mapping(result.get("hue_mapping")),
        "legal_ber": f"{result['legal_ber']:.6f}",
        "min_illegal_ber": "" if result.get("min_illegal_ber") is None else f"{float(result['min_illegal_ber']):.6f}",
        "raw_ber_at_min": "" if result.get("raw_ber_at_min") is None else f"{float(result['raw_ber_at_min']):.6f}",
        "average_illegal_ber": "" if result.get("average_illegal_ber") is None else f"{float(result['average_illegal_ber']):.6f}",
        "worst_illegal_position": result.get("worst_illegal_position") or "",
        "worst_legal_position": worst_legal_position,
        "worst_illegal_ber_vector": (
            ""
            if not result.get("worst_illegal_ber_vector")
            else "[" + ", ".join(f"{float(v):.6f}" for v in result["worst_illegal_ber_vector"]) + "]"
        ),
        "security_satisfied": "yes" if result.get("security_satisfied") else "no",
    }


def run_search() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _preload_matrices(project_root)
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)
    safe_probe_results_file = os.path.join(project_root, "test-4", SAFE_PROBE_RESULTS_FILENAME)
    existing_rows = load_existing_results(results_file)
    completed = {
        row["position_combination"]
        for row in existing_rows
        if row.get("security_satisfied") == "yes"
    }
    combinations = generate_position_combinations(project_root)
    print(f"Total combinations={len(combinations)}, safe_existing={len(completed)}")
    rng = random.Random(SELECTION_SEED)
    rows = existing_rows[:]

    for idx, legal_positions in enumerate(combinations, start=1):
        if str(tuple(legal_positions)) in completed:
            print(f"[{idx}/{len(combinations)}] Skip {legal_positions}: safe probes already exist.")
            continue

        print(f"[{idx}/{len(combinations)}] Searching combination {legal_positions}")
        result = search_security_aware_probes(project_root, legal_positions, rng)
        row = row_from_result(legal_positions, result)
        rows = [old for old in rows if old.get("position_combination") != str(tuple(legal_positions))]
        rows.append(row)
        if row["security_satisfied"] == "yes":
            completed.add(row["position_combination"])
        write_results(results_file, rows)
        write_safe_probe_results(safe_probe_results_file, rows)
        print(
            f"  New result: legal_ber={row['legal_ber']}, "
            f"min_illegal_ber={row['min_illegal_ber'] or 'N/A'}, "
            f"satisfied={row['security_satisfied']}"
        )

    print(f"Results saved to: {results_file}")
    return results_file


def main() -> None:
    run_search()


if __name__ == "__main__":
    main()
