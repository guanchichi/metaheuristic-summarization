"""Tests for reproducibility-related I/O helpers."""

import builtins
import random

import numpy as np
import pytest

from src.utils.io import set_global_seed


def test_global_seed_repeats_python_and_numpy_sequences():
    set_global_seed(2024)
    first = (random.random(), np.random.random())
    set_global_seed(2024)
    second = (random.random(), np.random.random())
    assert first == pytest.approx(second)


def test_broken_torch_import_is_not_silently_swallowed(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("broken torch installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="broken torch installation"):
        set_global_seed(2024)
