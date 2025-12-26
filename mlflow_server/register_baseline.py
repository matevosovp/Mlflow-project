import os
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine


@dataclass(frozen=True)
class Config:
    experiment_name: str = "model_FE_matevosov"
    registered_model_name: str = "real_estate_price_model_matevosov"
    run_name: str = "01_baseline_register"
    table_name: str = "public.real_estate_dataset_clean"
    target_col: str = "price"
    random_state: int = 42
    test_size: float = 0.2
    baseline_s3_uri: str = "s3://s3-student-mle-20251010-5a382f9c3d/models/real_estate/model.pkl"


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Environment variable {name} is required")
    return val


def get_db_uri() -> str:
    host = require_env("DB_DESTINATION_HOST")
    port = require_env("DB_DESTINATION_PORT")
    user = require_env("DB_DESTINATION_USER")
    password = require_env("DB_DESTINATION_PASSWORD")
    dbname = require_env("DB_DESTINATION_NAME")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def load_dataset(cfg: Config) -> pd.DataFrame:
    engine = create_engine(get_db_uri())
    query = f"SELECT * FROM {cfg.table_name}"
    df = pd.read_sql(query, engine)
    if cfg.target_col not in df.columns:
        raise RuntimeError(f"Target column '{cfg.target_col}' not found in table {cfg.table_name}")
    return df


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3 uri, got: {s3_uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download_from_s3(s3_uri: str, dst_path: Path) -> Path:
    endpoint = require_env("MLFLOW_S3_ENDPOINT_URL")
    aws_key = require_env("AWS_ACCESS_KEY_ID")
    aws_secret = require_env("AWS_SECRET_ACCESS_KEY")

    bucket, key = parse_s3_uri(s3_uri)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )
    s3.download_file(bucket, key, str(dst_path))
    return dst_path


def load_model(model_path: Path):
    # Пытаемся joblib, затем pickle
    try:
        return joblib.load(model_path)
    except Exception:
        pass

    with open(model_path, "rb") as f:
        return pickle.load(f)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def log_environment() -> None:
    # Логируем окружение максимально прозрачно
    import subprocess

    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        freeze = f"pip freeze failed: {e}"

    mlflow.log_text(freeze, artifact_file="environment/pip_freeze.txt")


def main() -> None:
    cfg = Config()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    df = load_dataset(cfg)
    y = df[cfg.target_col].to_numpy()
    X = df.drop(columns=[cfg.target_col])

    # Простейшая защита от полностью пустых колонок
    non_all_null_cols = [c for c in X.columns if not X[c].isna().all()]
    X = X[non_all_null_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    local_model_path = Path("tmp") / "baseline_model.pkl"
    download_from_s3(cfg.baseline_s3_uri, local_model_path)
    model = load_model(local_model_path)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.set_tag("stage", "baseline_registration")
        mlflow.log_param("baseline_s3_uri", cfg.baseline_s3_uri)
        mlflow.log_param("table_name", cfg.table_name)
        mlflow.log_param("target_col", cfg.target_col)
        mlflow.log_param("random_state", cfg.random_state)
        mlflow.log_param("test_size", cfg.test_size)
        mlflow.log_param("n_rows", int(df.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))

        # Метрики считаем на тестовой выборке
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, y_pred)
        mlflow.log_artifact(str(local_model_path), artifact_path="baseline_raw")

        # Регистрируем как MLflow Model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=cfg.registered_model_name,
        )

        log_environment()

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
        latest_version = max([int(v.version) for v in versions]) if versions else None

        print("Baseline registered")
        print(f"  experiment: {cfg.experiment_name}")
        print(f"  run_id: {run.info.run_id}")
        print(f"  registered_model_name: {cfg.registered_model_name}")
        print(f"  latest_version: {latest_version}")
        print(f"  model_uri: {model_info.model_uri}")


if __name__ == "__main__":
    main()
