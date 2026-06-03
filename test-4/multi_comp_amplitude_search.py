#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-component batch search with amplitude-aware hue mapping.

Strategy:
  1. For each 4-position combo, try to load existing zero-BER probe set
  2. If it doesn't work with multi-comp + amplitude-aware, search new probes
  3. Exact-eval: raw legal BER <= 0.03
  4. Security exact-eval: min illegal secure BER >= 0.2
  5. FEC confirm (10000 bits) to verify final reliability
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import sys
from typing import Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import test_4_simple as test

# 启用幅度感知
test.USE_AMPLITUDE_AWARE = True

LIGHT_CONDITION = "white"
OUTPUT_RESULTS_FILENAME = "multi_comp_amplitude_results_4.csv"

# 分阶段目标
RAW_LEGAL_TARGET = 0.03      # 放宽raw BER目标
MIN_ILLEGAL_TARGET = 0.2     # 安全性目标
FEC_LEGAL_TARGET = 0.005     # FEC纠正后目标

SEED = 20260509


def get_available_positions(project_root: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION)
    positions = []
    for entry in os.listdir(data_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() == ".csv" and stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def format_hue_mapping(hue_mapping: dict) -> str:
    if not hue_mapping:
        return ""
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(hue_mapping.items())) + "}"


def build_models(project_root: str, positions: Sequence[int], probes: np.ndarray):
    csv_files = test.build_csv_files_for_positions(project_root, positions, light_condition=LIGHT_CONDITION)
    matrices = test.load_selected_rows(csv_files, probes)
    models = [test.extract_fingerprint(probes, mat, force_positive_first=True) for mat in matrices]
    models = test.align_model_directions(models)
    hue_mapping = test.build_hue_mapping(
        models, probes,
        mapping_eval_bits=300, top_k_per_combination=3,
        use_amplitude_aware=True,
    )
    return models, hue_mapping


def exact_eval_legal(models: list, hue_mapping: dict) -> float:
    blocks = test.generate_all_bit_blocks(len(models))
    return test.evaluate_blocks_ber(models, blocks, hue_mapping)


def exact_eval_illegal(
    project_root: str,
    legal_positions: Sequence[int],
    probes: np.ndarray,
    legal_models: list,
    hue_mapping: dict,
) -> dict:
    legal_codes = [m.eff_code if m.eff_code is not None else m.code for m in legal_models]
    probe_to_row = test.build_probe_to_row(probes)
    blocks = test.generate_all_bit_blocks(len(legal_positions))
    all_positions = get_available_positions(project_root)

    global_min = float("inf")
    raw_at_min = 0.0
    worst_pos = None
    worst_idx = None
    all_secure = []

    for pos in all_positions:
        if pos in legal_positions:
            continue
        csv_file = os.path.join(project_root, "data", "15pro", LIGHT_CONDITION, f"{pos}.csv")
        mat = test.load_csv_matrix(csv_file)
        row_indices = [int((p / 5) - 1) for p in probes]
        illegal_mat = mat[row_indices].astype(float, copy=False)
        illegal_model = test.extract_fingerprint(probes, illegal_mat, force_positive_first=True)

        corr = test.calculate_correlation(
            legal_models[0].eff_code if legal_models[0].eff_code is not None else legal_models[0].code,
            illegal_model.eff_code if illegal_model.eff_code is not None else illegal_model.code
        )
        if corr < 0:
            if illegal_model.W is not None and illegal_model.W.ndim > 1:
                illegal_model.W = -illegal_model.W
                illegal_model.Z = -illegal_model.Z
                illegal_model.multi_code = -illegal_model.multi_code
                illegal_model.eff_w = -illegal_model.eff_w
                illegal_model.eff_code = -illegal_model.eff_code
            else:
                illegal_model.w = -illegal_model.w
                illegal_model.z = -illegal_model.z
                illegal_model.code = -illegal_model.code

        errors = np.zeros(len(legal_positions), dtype=float)
        totals = np.zeros(len(legal_positions), dtype=float)
        for bits_pm in blocks:
            _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
            hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
            obs = test.observe_block_from_measured_matrix(hue_seq, illegal_model.Y, probe_to_row)
            if illegal_model.eff_w is not None and illegal_model.eff_code is not None:
                dec = test.decode_local_block(obs, illegal_model.eff_w, illegal_model.eff_code)
            else:
                dec = test.decode_local_block(obs, illegal_model.w, illegal_model.code)
            true_bits = test.pm1_to_bin(bits_pm)
            for li in range(len(legal_positions)):
                totals[li] += 1
                if dec.bit_hat_bin != true_bits[li]:
                    errors[li] += 1

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

    return {
        "min_illegal_ber": float(global_min if all_secure else 0.0),
        "raw_ber_at_min": float(raw_at_min),
        "average_illegal_ber": float(np.mean(all_secure)) if all_secure else 0.0,
        "worst_illegal_position": worst_pos,
        "worst_legal_index": worst_idx,
    }


def fec_confirm(
    project_root: str,
    legal_positions: Sequence[int],
    probes: np.ndarray,
    legal_models: list,
    hue_mapping: dict,
    rng: random.Random,
) -> tuple[float, float]:
    legal_ber = test.evaluate_blocks_ber_with_convolutional_fec(
        legal_models,
        test.generate_random_information_bits(10000, len(legal_positions), rng=rng),
        hue_mapping,
    )
    if legal_ber > FEC_LEGAL_TARGET:
        return legal_ber, 0.0

    sec = exact_eval_illegal(project_root, legal_positions, probes, legal_models, hue_mapping)
    return legal_ber, sec["min_illegal_ber"]


def random_search_probes(
    project_root: str,
    legal_positions: Sequence[int],
    all_probes: list,
    rng: random.Random,
    max_samples: int = 500,
) -> tuple[np.ndarray, float, list, dict]:
    """随机搜索 probe sets，返回最佳的"""
    best_probes = None
    best_ber = float("inf")
    best_models = None
    best_mapping = None

    for i in range(max_samples):
        count = rng.choice([7, 8, 9, 10, 11, 12, 13])
        min_interval = 30 if count <= 12 else 20
        probes = np.sort(np.asarray(rng.sample(all_probes, count), dtype=float))
        if not test.is_valid_probe_set(probes, min_interval):
            continue

        try:
            models, hue_mapping = build_models(project_root, legal_positions, probes)
            ber = exact_eval_legal(models, hue_mapping)
            if ber < best_ber:
                best_ber = ber
                best_probes = probes.copy()
                best_models = models
                best_mapping = hue_mapping
                if i % 50 == 0:
                    print(f"    sample {i}: new best raw_ber={ber:.4f}, count={count}")
            if ber <= RAW_LEGAL_TARGET:
                print(f"    sample {i}: reached target! raw_ber={ber:.4f}")
                break
        except Exception:
            continue

    return best_probes, best_ber, best_models, best_mapping


def search_combination(project_root: str, positions: tuple[int, ...], rng: random.Random) -> dict:
    print(f"\n[Search] {positions}")
    all_probes = (5 + np.arange(73) * 5).astype(float).tolist()

    # Phase 1: search probes
    probes, raw_ber, models, hue_mapping = random_search_probes(
        project_root, positions, all_probes, rng, max_samples=120
    )

    if probes is None:
        print("  No probe candidates found!")
        return {"position_combination": str(positions), "probe_count": 0, "probes": "",
                "hue_mapping": "", "legal_ber": "1.0", "min_illegal_ber": "",
                "security_satisfied": "no"}

    print(f"  Best random probe set: count={len(probes)}, raw_ber={raw_ber:.6f}")

    if raw_ber > RAW_LEGAL_TARGET:
        print(f"  WARN: raw_ber {raw_ber:.4f} > {RAW_LEGAL_TARGET}, still trying security eval...")
        # 继续尝试security eval，记录实际表现

    # Phase 2: security eval
    sec = exact_eval_illegal(project_root, positions, probes, models, hue_mapping)
    print(f"  Security: min_illegal={sec['min_illegal_ber']:.4f}, worst_pos={sec['worst_illegal_position']}")

    if sec["min_illegal_ber"] < MIN_ILLEGAL_TARGET:
        print(f"  FAIL: security {sec['min_illegal_ber']:.4f} < {MIN_ILLEGAL_TARGET}")
        return {"position_combination": str(positions), "probe_count": len(probes),
                "probes": format_probes(probes), "hue_mapping": format_hue_mapping(hue_mapping),
                "legal_ber": f"{raw_ber:.6f}", "min_illegal_ber": f"{sec['min_illegal_ber']:.4f}",
                "security_satisfied": "no"}

    # Phase 3: FEC confirm
    print("  Running FEC confirmation...")
    fec_legal, fec_illegal = fec_confirm(project_root, positions, probes, models, hue_mapping, rng)
    print(f"  FEC: legal={fec_legal:.6f}, illegal={fec_illegal:.4f}")

    satisfied = fec_legal <= FEC_LEGAL_TARGET and fec_illegal >= MIN_ILLEGAL_TARGET
    print(f"  {'*** SECURITY SATISFIED ***' if satisfied else 'FAIL on FEC confirmation'}")

    return {
        "position_combination": str(positions),
        "probe_count": len(probes),
        "probes": format_probes(probes),
        "hue_mapping": format_hue_mapping(hue_mapping),
        "legal_ber": f"{fec_legal:.6f}",
        "min_illegal_ber": f"{fec_illegal:.4f}",
        "security_satisfied": "yes" if satisfied else "no",
    }


def write_results(results_file: str, rows: list[dict]) -> None:
    fieldnames = [
        "position_combination", "probe_count", "probes", "hue_mapping",
        "legal_ber", "min_illegal_ber", "security_satisfied",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, "test-4", OUTPUT_RESULTS_FILENAME)

    positions = get_available_positions(project_root)
    print(f"Total positions: {len(positions)}")
    print(f"Config: N_COMP={test.N_COMPONENTS}, ALPHA={test.ALPHA}")
    print(f"USE_AMPLITUDE_AWARE={test.USE_AMPLITUDE_AWARE}")
    print(f"Targets: raw_legal<={RAW_LEGAL_TARGET}, illegal>={MIN_ILLEGAL_TARGET}, FEC_legal<={FEC_LEGAL_TARGET}")

    # 只测一个已知组合，用大样本搜索
    combinations = [(1, 3, 12, 22)]
    print(f"Testing {len(combinations)} combinations (500 samples each)")

    rng = random.Random(SEED)
    rows = []

    for idx, pos in enumerate(combinations, start=1):
        print(f"\n[{idx}/{len(combinations)}] Processing {pos}")
        result = search_combination(project_root, pos, rng)
        rows.append(result)
        write_results(results_file, rows)
        print(f"  Result: legal={result['legal_ber']}, illegal={result.get('min_illegal_ber', 'N/A')}, sat={result['security_satisfied']}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
