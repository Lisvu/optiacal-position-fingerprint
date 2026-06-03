#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

from global_guarded_core import GlobalGuardedArgs, run_global_guarded_experiment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "mate40pro_high": r"data\mate40pro\high",
    "15pro_mid": r"data\15pro\mid",
}
K_VALUES = (2, 3, 7)


@dataclass(frozen=True)
class DatasetExperiment:
    dataset_name: str
    dataset_dir: str
    k: int


def default_experiments() -> list[DatasetExperiment]:
    return [
        DatasetExperiment(dataset_name, dataset_dir, k)
        for dataset_name, dataset_dir in DATASETS.items()
        for k in K_VALUES
    ]


def output_dir_for(base_dir: str, dataset_name: str, k: int) -> str:
    return os.path.join(base_dir, "dataset_compare", dataset_name, f"k{k}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global-guarded comparison on alternate datasets.")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_all(sample_size: int, overwrite: bool) -> list[tuple[DatasetExperiment, str]]:
    outputs: list[tuple[DatasetExperiment, str]] = []
    for experiment in default_experiments():
        output_dir = output_dir_for(BASE_DIR, experiment.dataset_name, experiment.k)
        summary_csv, _, _ = run_global_guarded_experiment(
            GlobalGuardedArgs(
                real_k=experiment.k,
                output_dir=output_dir,
                sample_size=sample_size,
                dataset_dir=experiment.dataset_dir,
                overwrite=overwrite,
            )
        )
        outputs.append((experiment, summary_csv))
        print(f"[OK] {experiment.dataset_name} k={experiment.k} -> {summary_csv}")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_all(sample_size=args.sample_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
