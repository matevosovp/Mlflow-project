"""Register a checksum-verified baseline model against the shared holdout split."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import boto3
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from mlflow_project.config import AppConfig
from mlflow_project.data import dataset_fingerprint, load_dataset, split_dataset
from mlflow_project.features import prepare_model_contract_frame
from mlflow_project.registry import update_registered_version
from mlflow_project.training import regression_metrics


def required_environment(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {name} is required")
    return value.strip()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("BASELINE_MODEL_S3_URI must be s3://bucket/key")
    return parsed.netloc, parsed.path.lstrip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_model(destination: Path) -> tuple[object, str]:
    uri = required_environment("BASELINE_MODEL_S3_URI")
    expected_sha256 = required_environment("BASELINE_MODEL_SHA256").lower()
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("BASELINE_MODEL_SHA256 must contain exactly 64 hex characters")

    bucket, key = parse_s3_uri(uri)
    client = boto3.client(
        "s3",
        endpoint_url=required_environment("MLFLOW_S3_ENDPOINT_URL"),
    )
    client.download_file(bucket, key, str(destination))
    actual_sha256 = sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Baseline checksum mismatch: refusing to deserialize the downloaded file"
        )

    # joblib/pickle deserialization can execute code. The mandatory trusted digest
    # is verified immediately before this call.
    return joblib.load(destination), actual_sha256


def baseline_features(model: object, frame):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        expected = [str(name) for name in feature_names]
        missing = sorted(set(expected) - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"Baseline model expects missing columns: {', '.join(missing)}"
            )
        return frame.loc[:, expected]

    drop_columns = tuple(
        value.strip()
        for value in os.getenv(
            "BASELINE_DROP_COLUMNS", "flat_id,building_id,studio"
        ).split(",")
        if value.strip()
    )
    return frame.drop(columns=list(drop_columns), errors="ignore")


def main() -> None:
    config = AppConfig.from_env()
    data = load_dataset(
        config.database,
        config.table_name,
        target_column=config.target_column,
    )
    splits = split_dataset(
        data,
        target_column=config.target_column,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_state,
        group_column=config.split_group_column,
    )

    with TemporaryDirectory() as directory:
        model, checksum = download_verified_model(Path(directory) / "baseline.joblib")

        X_test = baseline_features(model, splits.X_test)
        predictions = model.predict(X_test)
        metrics = regression_metrics(splits.y_test, predictions)

        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(run_name="register_verified_baseline") as run:
            mlflow.set_tags(
                {
                    "stage": "baseline",
                    "source": "checksum_verified_s3_object",
                    "split_policy": "shared_disjoint_holdout",
                }
            )
            mlflow.log_params(
                {
                    "table_name": config.table_name,
                    "target_column": config.target_column,
                    "random_state": config.random_state,
                    "baseline_sha256": checksum,
                    "dataset_sha256": dataset_fingerprint(data),
                    **splits.summary(),
                }
            )
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})

            input_example = prepare_model_contract_frame(
                X_test.head(5), drop_identifiers=False
            )
            signature = infer_signature(input_example, model.predict(input_example))
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=config.registered_model_name,
                serialization_format="cloudpickle",
            )

            version = update_registered_version(
                MlflowClient(),
                model_name=config.registered_model_name,
                run_id=run.info.run_id,
                tags={
                    "stage": "baseline",
                    "sha256": checksum,
                    "rmse_test": f"{metrics['rmse']:.6f}",
                },
                description=(
                    "Checksum-verified baseline evaluated on the shared test split. "
                    f"RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}. "
                    f"Run: {run.info.run_id}"
                ),
                alias="baseline",
            )

    print(
        json.dumps(
            {
                "run_id": run.info.run_id,
                "model_version": version,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
