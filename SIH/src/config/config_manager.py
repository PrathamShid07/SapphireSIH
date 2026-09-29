import os
from pathlib import Path
import yaml

def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

class ConfigManager:
    def __init__(self, config_path=None, root_dir=None):
        project_root = Path(__file__).resolve().parent.parent.parent
        
        if root_dir is None:
            self.root_dir = project_root
        else:
            self.root_dir = Path(root_dir).resolve()
            
        if config_path is None:
            default_in_root = self.root_dir / "config" / "default_config.yaml"
            if default_in_root.exists():
                self.config_path = default_in_root
            else:
                self.config_path = project_root / "config" / "default_config.yaml"
        else:
            self.config_path = Path(config_path).resolve()
            
        self.config = {}
        self.load_config()

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        self.config = self._load_with_extends(self.config_path, set())

    def _load_with_extends(self, path, seen):
        path = Path(path).resolve()
        if path in seen:
            raise ValueError(f"Circular 'extends' chain in configuration: {path}")
        seen = seen | {path}
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        parent = cfg.pop("extends", None)
        if parent is None:
            return cfg
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        if not parent_path.exists():
            raise FileNotFoundError(f"Base configuration '{parent}' extended by {path} not found at {parent_path}")
        return _deep_merge(self._load_with_extends(parent_path, seen), cfg)

    def get(self, key_path, default=None):
        keys = key_path.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def get_path(self, key_path):
        raw_path = self.get(key_path)
        if raw_path is None:
            return None
        p = Path(raw_path)
        if not p.is_absolute():
            return (self.root_dir / p).resolve()
        return p.resolve()

    def setup_directories(self):
        created_paths = []
        path_keys = [
            "paths.data_dir",
            "paths.input_dir",
            "paths.raw_dir",
            "paths.interim_dir",
            "paths.processed_dir",
            "paths.assets_dir",
            "paths.outputs_dir",
            "paths.logs_dir",
        ]
        for key in path_keys:
            p = self.get_path(key)
            if p:
                p.mkdir(parents=True, exist_ok=True)
                created_paths.append(p)
        return created_paths
