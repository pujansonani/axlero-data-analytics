"""Utility functions for configuration loading, path resolution, and reproducibility."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_rel_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration from repository root."""
    root = get_project_root()
    config_path = root / config_rel_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_path: str) -> Path:
    """Resolve a relative project path against the repository root."""
    return get_project_root() / rel_path


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds across Python and NumPy for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary to a JSON file with clean formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

