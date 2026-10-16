# SPDX-License-Identifier: Apache-2.0
"""Load the CPU kernels without importing Ray, FSDP, or rollout backends."""

import importlib.util
from pathlib import Path

import pytest


def load_source(relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cgl():
    return load_source("verl/trainer/ppo/cgl.py")


@pytest.fixture(scope="session")
def reward():
    return load_source("verl/utils/reward_score/android_control.py")
