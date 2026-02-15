from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_structured_config(path: str | Path) -> dict[str, Any]:
    """
    Load a dictionary config from JSON or YAML.

    YAML parsing requires PyYAML to be installed. We keep this optional so the
    base project can stay lightweight until training/eval scripts are needed.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    raw_text = cfg_path.read_text(encoding="utf-8")

    if suffix == ".json":
        loaded = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAML config requested but PyYAML is not installed. "
                "Install with `python -m pip install pyyaml` when ready."
            ) from exc
        loaded = yaml.safe_load(raw_text)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping/object: {cfg_path}")
    return loaded

