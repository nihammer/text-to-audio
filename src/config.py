"""Load and validate config.yaml settings."""
import os
from pathlib import Path
import yaml


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_config: dict | None = None


def load_config(config_path: str | None = None) -> dict:
    global _config
    if _config is not None:
        return _config

    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _config = _validate(data)
    return _config


def _validate(cfg: dict) -> dict:
    required_keys = ["reference", "tts", "audio", "chunker", "paths"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config section: '{key}'")

    tts = cfg["tts"]
    if tts.get("device") not in ("mps", "cpu"):
        raise ValueError(f"tts.device must be 'mps' or 'cpu', got: {tts.get('device')}")
    if not isinstance(tts.get("nfe_step"), int) or tts["nfe_step"] < 1:
        raise ValueError("tts.nfe_step must be a positive integer")
    if not isinstance(tts.get("speed"), (int, float)) or tts["speed"] <= 0:
        raise ValueError("tts.speed must be a positive number")

    chunker = cfg["chunker"]
    if not isinstance(chunker.get("max_chars"), int) or chunker["max_chars"] < 50:
        raise ValueError("chunker.max_chars must be an integer >= 50")

    return cfg


def get_config() -> dict:
    """Return cached config, loading it if necessary."""
    return load_config()
