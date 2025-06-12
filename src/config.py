import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    return yaml.safe_load(Path(path).read_text())
