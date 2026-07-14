"""Typed configuration loaded from environment variables and ``.env``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {name} is required")
    return value.strip()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        return cls(
            host=_required("DB_DESTINATION_HOST"),
            port=int(_required("DB_DESTINATION_PORT")),
            database=os.getenv("DB_DESTINATION_DBNAME")
            or _required("DB_DESTINATION_NAME"),
            user=_required("DB_DESTINATION_USER"),
            password=_required("DB_DESTINATION_PASSWORD"),
        )

    @property
    def url(self) -> URL:
        """Return an encoded SQLAlchemy URL without manual string interpolation."""

        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    table_name: str = "public.real_estate_dataset_clean"
    target_column: str = "price"
    split_group_column: str | None = "building_id"
    random_state: int = 42
    validation_size: float = 0.2
    test_size: float = 0.2
    reference_year: int = 2026
    optuna_trials: int = 10
    random_search_iterations: int = 10
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    def __post_init__(self) -> None:
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1")
        if self.optuna_trials < 1 or self.random_search_iterations < 1:
            raise ValueError("tuning iteration counts must be positive")

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> AppConfig:
        load_dotenv(dotenv_path or PROJECT_ROOT / ".env", override=False)
        return cls(
            database=DatabaseConfig.from_env(),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "model_improvement_matevosov"
            ),
            registered_model_name=os.getenv(
                "MLFLOW_REGISTERED_MODEL_NAME", "real_estate_price_model"
            ),
            table_name=os.getenv(
                "DATA_TABLE_NAME", "public.real_estate_dataset_clean"
            ),
            target_column=os.getenv("TARGET_COLUMN", "price"),
            split_group_column=(
                os.getenv("SPLIT_GROUP_COLUMN", "building_id").strip() or None
            ),
            random_state=_int_env("RANDOM_STATE", 42),
            validation_size=_float_env("VALIDATION_SIZE", 0.2),
            test_size=_float_env("TEST_SIZE", 0.2),
            reference_year=_int_env(
                "BUILD_YEAR_REFERENCE", datetime.now(UTC).year
            ),
            optuna_trials=_int_env("OPTUNA_TRIALS", 10),
            random_search_iterations=_int_env("RANDOM_SEARCH_ITERATIONS", 10),
            artifacts_dir=Path(
                os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
            ).resolve(),
        )
