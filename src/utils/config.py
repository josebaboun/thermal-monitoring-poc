"""Configuration management utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the thermal monitoring system."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def get(self, key: str, default=None):
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @property
    def video(self):
        """Get video configuration."""
        return self._config.get('video', {})

    @property
    def detection(self):
        """Get detection configuration."""
        return self._config.get('detection', {})

    @property
    def thermal(self):
        """Get thermal configuration."""
        return self._config.get('thermal', {})

    @property
    def visualization(self):
        """Get visualization configuration."""
        return self._config.get('visualization', {})

    @property
    def alerts(self):
        """Get alerts configuration."""
        return self._config.get('alerts', {})
