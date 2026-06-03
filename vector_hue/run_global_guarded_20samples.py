#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence

from global_guarded_core import GlobalGuardedArgs, run_global_guarded_experiment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_BASE = os.path.join(
    os.environ.get("TEMP", r"C:\Users\ASUS\AppData\Local\Temp"),
    "opencode",
    "global_guarded_20samples",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one global-guarded 20-sample experiment.")
    parser.add_argument("--k", type=int, required=True, choices=range(2, 11), help="real k value, from 2 to 10")
    parser.add_argument("--sample-size", type=int, default=20, help="number of random samples to run")
    return parser.parse_args(argv)


def final_summary_path(base_dir: str, k: int) -> str:
    return os.path.join(base_dir, f"k{k}", "global_guarded", "results_summary_20samples.csv")


def run_one(k: int, sample_size: int = 20) -> str:
    temp_dir = os.path.join(TEMP_BASE, f"k{k}")
    final_summary = final_summary_path(BASE_DIR, k)
    os.makedirs(os.path.dirname(final_summary), exist_ok=True)

    summary_csv, _, _ = run_global_guarded_experiment(
        GlobalGuardedArgs(
            real_k=k,
            output_dir=temp_dir,
            sample_size=sample_size,
            overwrite=True,
        )
    )
    shutil.copyfile(summary_csv, final_summary)
    return final_summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    final_summary = run_one(args.k, args.sample_size)
    print(f"[OK] k={args.k} -> {final_summary}")


if __name__ == "__main__":
    main()
