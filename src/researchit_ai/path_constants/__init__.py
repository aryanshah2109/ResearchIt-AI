from pathlib import Path

# Root path
ROOT_DIR = Path(__file__).resolve().parents[3]

# Config paths
CONFIG_DIR = ROOT_DIR / "src" / "researchit_ai" / "config"


# Log paths
LOG_PATH = ROOT_DIR / "logs"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB