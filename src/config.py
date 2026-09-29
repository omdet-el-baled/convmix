"""Small YAML inheritance/override layer; no Hydra global state or cwd changes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def merge_dict(base: dict, changes: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None,
                _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = project_path(path)
    if path in _stack:
        raise ValueError(f"Circular config inheritance: {_stack + (path,)}")
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    parents = values.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    result: dict = {}
    for parent in parents:
        # Parent config paths are relative to the file that includes them.
        result = merge_dict(result, load_config(path.parent / parent,
                                               _stack=_stack + (path,)))
    result = merge_dict(result, values)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Use key=value, got {override!r}")
        key, text = override.split("=", 1)
        set_value(result, key, yaml.safe_load(text))
    return result


def set_value(config: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise KeyError(f"Unknown configuration key: {key}")
        current = current[part]
    if parts[-1] not in current:
        raise KeyError(f"Unknown configuration key: {key}")
    current[parts[-1]] = value


def config_hash(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_config(config: dict) -> None:
    for name in ("data", "sde", "network", "model", "trainer", "sampling", "evaluation", "run"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing config section {name!r}")
    if config["data"]["num_sources"] < 2:
        raise ValueError("This package is the single-sensor K>=2 problem.")
    if not 0 < config["model"]["t_eps"] < config["sde"]["T"]:
        raise ValueError("Require 0 < model.t_eps < sde.T")
    if not 0 <= config["model"]["mismatch_probability"] <= 1:
        raise ValueError("mismatch_probability must be in [0,1]")
    if not config.get("components") and config["sampling"]["solver"] not in {"euler", "heun", "rk4", "em", "pc"}:
        raise ValueError("Unknown sampling.solver")
    if config["sampling"]["num_steps"] < 1:
        raise ValueError("sampling.num_steps must be positive")
    # Complex SDE calculations stay complex64. Allow bfloat16 network autocast,
    # but not true fp16/fp64 casting of the entire Lightning module.
    if config["trainer"]["precision"] not in {"32-true", "bf16-mixed"}:
        raise ValueError("Supported precision: 32-true or bf16-mixed (test on your GPU).")
    if config["trainer"]["devices"] != 1:
        raise ValueError("This packaged workflow currently supports one CPU/GPU device.")
    if config["trainer"]["max_epochs"] < 1:
        raise ValueError("trainer.max_epochs must be positive")
