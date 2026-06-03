#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from global_guarded_core import run_cli


if __name__ == "__main__":
    run_cli(7, os.path.dirname(os.path.abspath(__file__)))
