#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from global_guarded_core import GlobalGuardedArgs, run_global_guarded_experiment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = r"E:\LuminaLink\Position_fingerprint_experiment\data\p40\low_column_shuffled"
DATASET_NAME = "p40_low_column_shuffled"
K_VALUES = tuple(range(2, 21))


def output_dir_for(base_dir: str, k: int) -> str:
    return os.path.join(base_dir, "dataset_compare", DATASET_NAME, f"k{k}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run p40 low_column_shuffled global-guarded batch experiments.")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_all(sample_size: int = 5, overwrite: bool = False) -> list[str]:
    outputs: list[str] = []
    for k in K_VALUES:
        summary_csv, _, _ = run_global_guarded_experiment(
            GlobalGuardedArgs(
                real_k=k,
                output_dir=output_dir_for(BASE_DIR, k),
                sample_size=sample_size,
                dataset_dir=DATASET_DIR,
                overwrite=overwrite,
            )
        )
        outputs.append(summary_csv)
        print(f"[OK] {DATASET_NAME} k={k} -> {summary_csv}")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_all(sample_size=args.sample_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
