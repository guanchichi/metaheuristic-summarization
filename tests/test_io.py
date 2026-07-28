"""Tests for reproducibility-related I/O helpers."""

import builtins
from io import StringIO
import random

import numpy as np
import pytest

from src.utils.io import set_global_seed, write_jsonl


def test_jsonl_writer_pins_lf_newlines(monkeypatch):
    captured = {}

    class NonClosingBuffer(StringIO):
        def __exit__(self, *_args):
            return False

    buffer = NonClosingBuffer()

    def fake_open(_path, _mode, **kwargs):
        captured.update(kwargs)
        return buffer

    monkeypatch.setattr("src.utils.io.ensure_dir", lambda _path: None)
    monkeypatch.setattr("builtins.open", fake_open)
    write_jsonl("artifact.jsonl", [{"id": "row"}])

    assert captured["newline"] == "\n"
    assert buffer.getvalue() == '{"id": "row"}\n'


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
