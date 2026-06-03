#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import run_mate40pro_high_column_shuffled as runner


def test_parse_args_accepts_requested_k_values() -> None:
    args = runner.parse_args(["--k", "8"])
    assert args.k == 8
    assert args.sample_size == 3
    assert args.overwrite is True


def test_output_dir_points_to_dataset_compare_folder() -> None:
    path = runner.output_dir_for(r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue", 7)
    assert path == os.path.join(
        r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue",
        "dataset_compare",
        "mate40pro_high_column_shuffled",
        "k7",
    )


if __name__ == "__main__":
    test_parse_args_accepts_requested_k_values()
    test_output_dir_points_to_dataset_compare_folder()
    print("mate40pro high column shuffled runner checks passed")
