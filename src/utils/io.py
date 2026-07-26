import os
import json
import yaml
import datetime as _dt
import tempfile
from typing import Any, Dict, Iterable, Optional


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL atomically so a failed preprocessor leaves no partial artifact."""

    output_dir = os.path.dirname(path) or "."
    ensure_dir(output_dir)
    handle, temporary_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".partial",
        dir=output_dir,
        text=True,
    )
    os.close(handle)
    try:
        write_jsonl(temporary_path, rows)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        import random, numpy as np

        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    except Exception:
        pass
