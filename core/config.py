"""
configuration loader. YAML config with CLI overrides.
"""

import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def load(config_path=None):
    """load config from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def override(config, overrides):
    """apply CLI overrides to config. format: 'llm.provider=openrouter'"""
    for item in overrides:
        if "=" not in item:
            continue
        key_path, value = item.split("=", 1)
        keys = key_path.split(".")

        # try to cast to appropriate type
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        target = config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    return config


def get_llm_kwargs(config):
    """extract provider-specific kwargs from config."""
    llm = config["llm"]
    provider = llm["provider"]
    kwargs = {"model": llm["model"]}

    # provider-specific settings
    if provider == "ollama" and "ollama" in llm:
        kwargs["host"] = llm["ollama"].get("host", "http://localhost:11434")

    return provider, kwargs
