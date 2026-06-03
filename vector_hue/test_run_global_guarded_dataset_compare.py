#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import run_global_guarded_dataset_compare as runner


def test_default_experiments_cover_two_datasets_and_three_k_values() -> None:
    experiments = runner.default_experiments()
    assert len(experiments) == 6
    assert {(experiment.dataset_name, experiment.k) for experiment in experiments} == {
        ("mate40pro_high", 2),
        ("mate40pro_high", 3),
        ("mate40pro_high", 7),
        ("15pro_mid", 2),
        ("15pro_mid", 3),
        ("15pro_mid", 7),
    }


def test_output_dir_uses_dataset_name_and_k() -> None:
    path = runner.output_dir_for(r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue", "15pro_mid", 7)
    assert path == os.path.join(
        r"E:\LuminaLink\Position_fingerprint_experiment\vector_hue",
        "dataset_compare",
        "15pro_mid",
        "k7",
    )


if __name__ == "__main__":
    test_default_experiments_cover_two_datasets_and_three_k_values()
    test_output_dir_uses_dataset_name_and_k()
    print("dataset compare runner checks passed")
