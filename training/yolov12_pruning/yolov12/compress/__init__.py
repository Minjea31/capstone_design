"""Shared project pruning implementation, without patching installed PyTorch."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "capstone_pruning", Path(__file__).resolve().parents[3] / "pruning.py",
)
prune = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prune
_spec.loader.exec_module(prune)
