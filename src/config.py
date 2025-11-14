import yaml
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

class Config:
    def __init__(self, path: str = None):
        cfg_path = path or DEFAULT_CONFIG_PATH
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def __getitem__(self, item):
        return self._cfg[item]
