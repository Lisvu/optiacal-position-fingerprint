#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import run_mate40pro_high_column_shuffled_batch as mate40_batch
import run_p40_low_column_shuffled_batch as p40_batch


def test_p40_batch_defaults() -> None:
    args = p40_batch.parse_args([])
    assert args.sample_size == 5
    assert args.overwrite is False
    assert p40_batch.K_VALUES == tuple(range(2, 21))
    assert p40_batch.DATASET_NAME == "p40_low_column_shuffled"


def test_mate40pro_batch_defaults() -> None:
    args = mate40_batch.parse_args([])
    assert args.sample_size == 5
    assert args.overwrite is False
    assert mate40_batch.K_VALUES == tuple(range(2, 21))
    assert mate40_batch.DATASET_NAME == "mate40pro_high_column_shuffled"


def test_batch_output_paths() -> None:
    base = r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue"
    assert p40_batch.output_dir_for(base, 20) == os.path.join(
        base,
        "dataset_compare",
        "p40_low_column_shuffled",
        "k20",
    )
    assert mate40_batch.output_dir_for(base, 2) == os.path.join(
        base,
        "dataset_compare",
        "mate40pro_high_column_shuffled",
        "k2",
    )


if __name__ == "__main__":
    test_p40_batch_defaults()
    test_mate40pro_batch_defaults()
    test_batch_output_paths()
    print("column shuffled batch runner checks passed")
