# core/settings.py
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from dotenv import load_dotenv

load_dotenv()  # .env 로드 (DATABASE_URL 등)

# ----- 설정 스키마 (타입 명시) -----
@dataclass
class ArtifactsCfg:
    root: str
    project: str
    models_dir: str = "models"
    plots_dir: str = "plots"
    meta_dir: str = "meta"

    def as_paths(self) -> Dict[str, Path]:
        base = Path(self.root) / self.project
        return {
            "base": base,
            "models": base / self.models_dir,
            "plots": base / self.plots_dir,
            "meta": base / self.meta_dir,
        }

@dataclass
class SavingCfg:
    save_best_only: bool = True
    best_filename: str = "best.safetensors"
    best_hparams_filename: str = "best_hp.json"

@dataclass
class LoggingCfg:
    table: str = "training_logs"
    experiment: str = "default"

@dataclass
class DatabaseCfg:
    url: str  # DATABASE_URL (env 우선)

@dataclass
class Settings:
    artifacts: ArtifactsCfg
    saving: SavingCfg
    logging: LoggingCfg
    database: DatabaseCfg

# ----- 로더 -----
def load_settings(yaml_path: str = "configs/default.yaml") -> Settings:
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)

    # DATABASE_URL은 .env가 우선, 없으면 yaml의 database.url 사용(있다면)
    db_url_env = os.getenv("DATABASE_URL")
    db_url_yaml = (y.get("database", {}) or {}).get("url")
    if not db_url_env and not db_url_yaml:
        raise RuntimeError("DATABASE_URL not found in environment or configs/default.yaml")

    artifacts = ArtifactsCfg(**y["artifacts"])
    saving = SavingCfg(**(y.get("saving") or {}))
    logging = LoggingCfg(**(y.get("logging") or {}))
    database = DatabaseCfg(url=db_url_env or db_url_yaml)

    return Settings(
        artifacts=artifacts,
        saving=saving,
        logging=logging,
        database=database,
    )
