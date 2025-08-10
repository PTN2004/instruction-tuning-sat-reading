import yaml
import os


def load_config(config_name: str) -> dict:

    config_path = os.path.join("configs", f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
