"""Compatibility entry point for canonical Multi-News preprocessing.

Use ``python -m src.data.preprocess_multinews --help`` for the authoritative
interface.  Keeping this wrapper avoids leaving the old boundary-destroying
implementation at a prominent repository path.
"""

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess_multinews import main


if __name__ == "__main__":
    main()
