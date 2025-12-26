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
    registered_model_name: str = "real_estate_price_model"
    run_name: str = "01_baseline"
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

    
    run_name = cfg.run_name

    with mlflow.start_run(run_name=run_name) as run:
        # Теги на уровне run (видно в Experiments)
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("run_purpose", "register_baseline_model")
        mlflow.set_tag("model_lineage", "baseline_from_s3")
        mlflow.set_tag("data_table", cfg.table_name)
        mlflow.set_tag("target", cfg.target_col)

        # Параметры
        mlflow.log_param("model_kind", "baseline")
        mlflow.log_param("baseline_s3_uri", cfg.baseline_s3_uri)
        mlflow.log_param("table_name", cfg.table_name)
        mlflow.log_param("target_col", cfg.target_col)
        mlflow.log_param("random_state", cfg.random_state)
        mlflow.log_param("test_size", cfg.test_size)
        mlflow.log_param("n_rows", int(df.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))

        # Метрики baseline считаем на тестовой выборке
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, y_pred)

        # Сохраняем "сырую" baseline модель как артефакт (как была скачана)
        mlflow.log_artifact(str(local_model_path), artifact_path="baseline/raw_model")

        # Логируем модель в MLflow + регистрируем в Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=cfg.registered_model_name,
        )

        # Окружение
        log_environment()

        # Помечаем именно ВЕРСИЮ в Model Registry, чтобы в UI было очевидно
        client = MlflowClient()

        # Находим версию, созданную именно этим run
        versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
        versions_this_run = [v for v in versions if getattr(v, "run_id", None) == run.info.run_id]

        if not versions_this_run:
            raise RuntimeError(
                "Не удалось найти model version, созданную текущим run. "
                "Проверьте, что registered_model_name задан верно."
            )

        # Берем самую свежую версию текущего run
        new_v = sorted(versions_this_run, key=lambda v: int(v.version))[-1]
        new_version = str(new_v.version)  # ВАЖНО: строка

        # Теги на уровне версии (Model Registry)
        client.set_model_version_tag(cfg.registered_model_name, new_version, "stage", "baseline")
        client.set_model_version_tag(cfg.registered_model_name, new_version, "source", "s3")
        client.set_model_version_tag(cfg.registered_model_name, new_version, "baseline_s3_uri", cfg.baseline_s3_uri)
        client.set_model_version_tag(cfg.registered_model_name, new_version, "table_name", cfg.table_name)
        client.set_model_version_tag(cfg.registered_model_name, new_version, "target_col", cfg.target_col)

        # Метрики лучше писать как строки
        client.set_model_version_tag(cfg.registered_model_name, new_version, "rmse_test", f"{metrics.get('rmse', float('nan')):.6f}")
        client.set_model_version_tag(cfg.registered_model_name, new_version, "mae_test", f"{metrics.get('mae', float('nan')):.6f}")
        client.set_model_version_tag(cfg.registered_model_name, new_version, "r2_test", f"{metrics.get('r2', float('nan')):.6f}")

        desc = (
            "Этап 1. Baseline модель (регистрация)\n"
            f"- Источник: {cfg.baseline_s3_uri}\n"
            f"- Данные: {cfg.table_name}, target={cfg.target_col}\n"
            f"- Метрики на test: RMSE={metrics.get('rmse'):.6f}, MAE={metrics.get('mae'):.6f}, R2={metrics.get('r2'):.6f}\n"
            f"- Run: {run.info.run_id}\n"
        )

        # update_model_version тоже принимает version как str
        client.update_model_version(
            name=cfg.registered_model_name,
            version=new_version,
            description=desc,
        )

        print("Baseline registered")
        print(f"  experiment: {cfg.experiment_name}")
        print(f"  run_id: {run.info.run_id}")
        print(f"  registered_model_name: {cfg.registered_model_name}")
        print(f"  model_version: {new_version}")
        print(f"  model_uri: {model_info.model_uri}")



if __name__ == "__main__":
    main()
