"""Configuration settings"""

from pathlib import Path

# ROOT_DIR = Path(__file__).parent.parent
# ROOT_DIR = Path(__file__).parent.parent
ROOT_DIR = Path("/lustre") / Path.home().name / "ml-notebook"
ROOT_DIR.mkdir(parents=True, exist_ok=True)


DATA_DIR = ROOT_DIR / "data"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_CACHE = DATA_DIR / "cache"
DATA_CUSTOM = DATA_DIR / "custom"
RUNS_DIR = ROOT_DIR / "runs"
WEIGHTS_DIR = ROOT_DIR / "weights"
