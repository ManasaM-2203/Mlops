import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)
