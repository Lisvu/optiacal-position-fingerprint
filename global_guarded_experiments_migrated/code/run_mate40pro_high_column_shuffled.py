#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from global_guarded_core import GlobalGuardedArgs, run_global_guarded_experiment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = r"E:\LuminaLink\Position_fingerprint_experiment\data\mate40pro\high_column_shuffled"
DATASET_NAME = "mate40pro_high_column_shuffled"
K_VALUES = (2, 3, 4, 7, 8)


def output_dir_for(base_dir: str, k: int) -> str:
    return os.path.join(base_dir, "dataset_compare", DATASET_NAME, f"k{k}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global-guarded experiment on mate40pro high_column_shuffled.")
    parser.add_argument("--k", type=int, required=True, choices=K_VALUES)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="continue existing output instead of overwriting it")
    args = parser.parse_args(argv)
    args.overwrite = not args.resume
    return args


def run_one(k: int, sample_size: int = 3, overwrite: bool = True) -> str:
    summary_csv, _, _ = run_global_guarded_experiment(
        GlobalGuardedArgs(
            real_k=k,
            output_dir=output_dir_for(BASE_DIR, k),
            sample_size=sample_size,
            dataset_dir=DATASET_DIR,
            overwrite=overwrite,
        )
    )
    return summary_csv


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_csv = run_one(args.k, sample_size=args.sample_size, overwrite=args.overwrite)
    print(f"[OK] k={args.k} -> {summary_csv}")


if __name__ == "__main__":
    main()
