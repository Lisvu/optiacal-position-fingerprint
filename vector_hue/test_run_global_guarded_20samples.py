#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import run_global_guarded_20samples as runner


def test_final_summary_path_points_to_requested_k_directory() -> None:
    path = runner.final_summary_path(r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue", 7)
    assert path == os.path.join(
        r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue",
        "k7",
        "global_guarded",
        "results_summary_20samples.csv",
    )


def test_parse_args_accepts_single_k() -> None:
    args = runner.parse_args(["--k", "10"])
    assert args.k == 10
    assert args.sample_size == 20


if __name__ == "__main__":
    test_final_summary_path_points_to_requested_k_directory()
    test_parse_args_accepts_single_k()
    print("global guarded 20samples runner checks passed")
