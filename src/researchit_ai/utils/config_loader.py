from pathlib import Path
from functools import lru_cache
import yaml

from src.researchit_ai.path_constants import CONFIG_DIR


class ConfigLoader:
    """Loads application configuration from YAML files."""

    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir

        self.orchestrator = self._load("orchestrator_config.yaml")
        self.tool = self._load("tool_config.yaml")
        self.rag = self._load("rag_config.yaml")
        self.agent = self._load("agent_config.yaml")
        self.evals = self._load("evals_config.yaml")

    def _load(self, filename: str) -> dict:
        """Load a single YAML configuration file."""

        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

@lru_cache(maxsize=1)
def get_config() -> ConfigLoader:
    """Process-wide singleton so YAML is read once."""
    return ConfigLoader()