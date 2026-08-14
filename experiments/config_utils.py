"""Load experiment configs from config/*.yaml."""
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent.parent / 'config'

def load_config(name):
    """name: config filename (e.g. 'benchmark_d100.yaml') or full path."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / name
    with open(path) as fh:
        return yaml.safe_load(fh)
