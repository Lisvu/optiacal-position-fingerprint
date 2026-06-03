#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from virtual_stream_core import ExperimentConfig, run_k_experiment


if __name__ == "__main__":
    run_k_experiment(ExperimentConfig(real_k=7, output_dir=os.path.dirname(os.path.abspath(__file__))))
