#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全性测试脚本。

目标：
1. 从 random_20_batch_test_results.csv 读取合法三位置组合和对应探针。
2. 对每组合法位置，使用其三个合法位置建立发送与映射模型。
3. 遍历所有不在该组合中的非法位置。
4. 非法位置仅使用自己的指纹信息和同一组探针进行解码。
5. 输出每个非法位置对三个合法位置的 BER、平均 BER，以及累计 BER。
"""

from __future__ import annotations

import ast
import csv
import os
import random
import sys
import types
from typing import Sequence

import numpy as np

LIGHT_CONDITION = "white"
SOURCE_RESULTS_FILENAME = "random_20_batch_test_results.csv"
OUTPUT_RESULTS_FILENAME = "security_illegal_position_results.csv"
NUM_BITS = 10000
MAPPING_EVAL_BITS = 500
MAPPING_TOP_K = 3
BASE_SEED = 20260401


def load_test_module() -> types.ModuleType:
    module_name = "test_3_simple_runtime"
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_3_simple.py")
    with open(module_path, "r", encoding="utf-8-sig") as f:
        source = f.read().lstrip("\ufeff")

    module = types.ModuleType(module_name)
    module.__file__ = module_path
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


test = load_test_module()


def parse_position_combination(text: str) -> tuple[int, int, int]:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"无效的位置组合: {text}")
    return tuple(int(v) for v in value)


def parse_probes(text: str) -> list[float]:
    value = ast.literal_eval(text)
    if not isinstance(value, list) or not value:
        raise ValueError(f"无效的探针列表: {text}")
    return [float(v) for v in value]


def format_probes(probes: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.1f}" for v in probes) + "]"


def load_probe_configs(source_file: str) -> list[dict]:
    with open(source_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "position_combination": parse_position_combination(row["position_combination"]),
                "best_probe_count": int(row["best_probe_count"]),
                "best_probes": parse_probes(row["best_probes"]),
                "best_ber": float(row["best_ber"]),
                "test_ber": float(row["test_ber"]),
            })
    return rows


def get_available_positions(project_root: str, light_condition: str) -> list[int]:
    data_dir = os.path.join(project_root, "data", "15pro", light_condition)
    positions = []
    for entry in os.listdir(data_dir):
        if not entry.endswith(".csv"):
            continue
        stem = os.path.splitext(entry)[0]
        if stem.isdigit():
            positions.append(int(stem))
    return sorted(positions)


def build_legal_context(
    project_root: str,
    legal_positions: Sequence[int],
    probes: Sequence[float],
    rng: random.Random,
) -> tuple[list[test.FingerprintModel], dict[tuple[int, ...], int], list[np.ndarray]]:
    csv_files = test.build_csv_files_for_positions(
        project_root,
        legal_positions,
        light_condition=LIGHT_CONDITION,
    )
    models, hue_mapping = test.build_models_from_probes(
        csv_files=csv_files,
        probes=np.asarray(probes, dtype=float),
        mapping_eval_bits=MAPPING_EVAL_BITS,
        mapping_top_k=MAPPING_TOP_K,
        rng=rng,
    )
    bit_blocks_pm = test.generate_random_bit_blocks(NUM_BITS, len(legal_positions), rng=rng)
    return models, hue_mapping, bit_blocks_pm


def evaluate_illegal_position(
    project_root: str,
    legal_models: Sequence[test.FingerprintModel],
    hue_mapping: dict[tuple[int, ...], int],
    illegal_position: int,
    probes: Sequence[float],
    bit_blocks_pm: Sequence[np.ndarray],
) -> tuple[list[float], float]:
    illegal_csv_file = test.build_csv_files_for_positions(
        project_root,
        [illegal_position],
        light_condition=LIGHT_CONDITION,
    )[0]
    probes_array = np.asarray(probes, dtype=float)
    illegal_matrix = test.load_selected_rows([illegal_csv_file], probes_array)[0]
    illegal_model = test.extract_fingerprint(probes_array, illegal_matrix, force_positive_first=True)

    probe_to_row = test.build_probe_to_row(probes_array)
    legal_codes = [model.code for model in legal_models]
    position_errors = np.zeros(len(legal_models), dtype=float)
    position_total = np.zeros(len(legal_models), dtype=float)

    for bits_pm in bit_blocks_pm:
        bits_pm = np.asarray(bits_pm, dtype=int)
        _, symbol_combinations = test.build_symbol_sequence(bits_pm, legal_codes)
        hue_seq = test.map_symbol_to_hue(symbol_combinations, hue_mapping)
        illegal_observation = test.observe_block_from_measured_matrix(hue_seq, illegal_matrix, probe_to_row)
        illegal_dec = test.decode_local_block(illegal_observation, illegal_model.w, illegal_model.code)
        true_bits = test.pm1_to_bin(bits_pm)

        for idx in range(len(legal_models)):
            position_total[idx] += 1
            if illegal_dec.bit_hat_bin != true_bits[idx]:
                position_errors[idx] += 1

    position_bers = (position_errors / np.maximum(position_total, 1.0)).tolist()
    average_ber = float(np.mean(position_bers)) if position_bers else 0.0
    return [float(v) for v in position_bers], average_ber


def write_results(results_file: str, rows: Sequence[dict]) -> None:
    fieldnames = [
        "legal_position_combination",
        "illegal_position",
        "best_probe_count",
        "test_probes",
        "source_best_ber",
        "source_test_ber",
        "num_bits",
        "ber_vs_legal_pos_1",
        "ber_vs_legal_pos_2",
        "ber_vs_legal_pos_3",
        "average_ber",
        "cumulative_ber",
    ]
    with open(results_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_security_validation() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(project_root, "test-3", SOURCE_RESULTS_FILENAME)
    results_file = os.path.join(project_root, "test-3", OUTPUT_RESULTS_FILENAME)

    configs = load_probe_configs(source_file)
    available_positions = get_available_positions(project_root, LIGHT_CONDITION)
    result_rows = []
    cumulative_average_sum = 0.0
    cumulative_average_count = 0

    for config_idx, config in enumerate(configs, start=1):
        legal_positions = config["position_combination"]
        probes = config["best_probes"]
        illegal_positions = [pos for pos in available_positions if pos not in legal_positions]
        rng = random.Random(BASE_SEED + config_idx)

        print(f"[{config_idx}/{len(configs)}] 开始处理合法位置组合 {legal_positions}，探针 {probes}")
        legal_models, hue_mapping, bit_blocks_pm = build_legal_context(
            project_root=project_root,
            legal_positions=legal_positions,
            probes=probes,
            rng=rng,
        )

        for illegal_position in illegal_positions:
            position_bers, average_ber = evaluate_illegal_position(
                project_root=project_root,
                legal_models=legal_models,
                hue_mapping=hue_mapping,
                illegal_position=illegal_position,
                probes=probes,
                bit_blocks_pm=bit_blocks_pm,
            )

            cumulative_average_count += 1
            cumulative_average_sum += average_ber
            cumulative_ber = cumulative_average_sum / cumulative_average_count

            print(
                f"  非法位置 {illegal_position}: "
                f"BER=({position_bers[0]:.6f}, {position_bers[1]:.6f}, {position_bers[2]:.6f}), "
                f"平均BER={average_ber:.6f}, 累计BER={cumulative_ber:.6f}"
            )

            result_rows.append({
                "legal_position_combination": str(tuple(legal_positions)),
                "illegal_position": illegal_position,
                "best_probe_count": config["best_probe_count"],
                "test_probes": format_probes(probes),
                "source_best_ber": f"{config['best_ber']:.6f}",
                "source_test_ber": f"{config['test_ber']:.6f}",
                "num_bits": NUM_BITS,
                "ber_vs_legal_pos_1": f"{position_bers[0]:.6f}",
                "ber_vs_legal_pos_2": f"{position_bers[1]:.6f}",
                "ber_vs_legal_pos_3": f"{position_bers[2]:.6f}",
                "average_ber": f"{average_ber:.6f}",
                "cumulative_ber": f"{cumulative_ber:.6f}",
            })

    write_results(results_file, result_rows)
    print(f"结果已保存到: {results_file}")
    return results_file


def main() -> None:
    run_security_validation()


if __name__ == "__main__":
    main()
