"""Leak-free tuning, final evaluation and MLflow registration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from .config import PROJECT_ROOT, AppConfig
from .data import DatasetSplits, dataset_fingerprint, load_dataset, split_dataset
from .features import build_model_pipeline, prepare_model_contract_frame
from .registry import update_registered_version


@dataclass(frozen=True)
class CandidateResult:
    method: str
    parameters: dict[str, Any]
    validation_rmse: float
    validation_mae: float
    validation_r2: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    model_version: str
    selected_method: str
    selected_parameters: dict[str, Any]
    test_rmse: float
    test_mae: float
    test_r2: float


def regression_metrics(
    y_true: pd.Series | np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _fit_and_score(
    splits: DatasetSplits,
    config: AppConfig,
    parameters: dict[str, Any],
    method: str,
    elapsed_seconds: float,
) -> CandidateResult:
    pipeline = build_model_pipeline(
        splits.X_train,
        reference_year=config.reference_year,
        random_state=config.random_state,
        model_params=parameters,
    )
    pipeline.fit(splits.X_train, splits.y_train)
    predictions = pipeline.predict(splits.X_validation)
    metrics = regression_metrics(splits.y_validation, predictions)
    return CandidateResult(
        method=method,
        parameters=parameters,
        validation_rmse=metrics["rmse"],
        validation_mae=metrics["mae"],
        validation_r2=metrics["r2"],
        elapsed_seconds=elapsed_seconds,
    )


def tune_with_optuna(splits: DatasetSplits, config: AppConfig) -> CandidateResult:
    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "iterations": trial.suggest_categorical("iterations", [400, 600, 800, 1100]),
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.16, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 40.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.7, 1.0
            ),
            "feature_percentile": trial.suggest_categorical(
                "feature_percentile", [50, 75, 100]
            ),
        }
        pipeline = build_model_pipeline(
            splits.X_train,
            reference_year=config.reference_year,
            random_state=config.random_state,
            model_params=parameters,
        )
        pipeline.fit(splits.X_train, splits.y_train)
        predictions = pipeline.predict(splits.X_validation)
        return float(root_mean_squared_error(splits.y_validation, predictions))

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    started = time.perf_counter()
    study.optimize(objective, n_trials=config.optuna_trials)
    elapsed = time.perf_counter() - started
    return _fit_and_score(
        splits,
        config,
        dict(study.best_params),
        method="optuna",
        elapsed_seconds=elapsed,
    )


def make_predefined_search_data(
    splits: DatasetSplits,
) -> tuple[pd.DataFrame, pd.Series, PredefinedSplit]:
    """Use train for fitting and the separate validation set for every candidate."""

    X_search = pd.concat(
        [splits.X_train, splits.X_validation], axis=0, ignore_index=True
    )
    y_search = pd.concat(
        [splits.y_train, splits.y_validation], axis=0, ignore_index=True
    )
    fold = np.concatenate(
        [
            np.full(len(splits.X_train), -1, dtype=int),
            np.zeros(len(splits.X_validation), dtype=int),
        ]
    )
    return X_search, y_search, PredefinedSplit(test_fold=fold)


def _normalize_search_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        if key.startswith("model__"):
            normalized[key.removeprefix("model__")] = value
        elif key == "select__percentile":
            normalized["feature_percentile"] = value
        else:
            raise ValueError(f"Unexpected search parameter: {key}")
    return normalized


def tune_with_random_search(
    splits: DatasetSplits, config: AppConfig
) -> CandidateResult:
    pipeline = build_model_pipeline(
        splits.X_train,
        reference_year=config.reference_year,
        random_state=config.random_state,
    )
    X_search, y_search, predefined_split = make_predefined_search_data(splits)
    parameter_distributions = {
        "model__iterations": [400, 600, 800, 1100],
        "model__depth": [5, 6, 8, 10],
        "model__learning_rate": [0.04, 0.06, 0.08, 0.1, 0.14],
        "model__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0, 20.0, 40.0],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bylevel": [0.7, 0.8, 0.9, 1.0],
        "select__percentile": [50, 75, 100],
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=config.random_search_iterations,
        scoring="neg_root_mean_squared_error",
        cv=predefined_split,
        refit=False,
        n_jobs=1,
        random_state=config.random_state,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X_search, y_search)
    elapsed = time.perf_counter() - started
    return _fit_and_score(
        splits,
        config,
        _normalize_search_parameters(dict(search.best_params_)),
        method="random_search",
        elapsed_seconds=elapsed,
    )


def train_and_register(
    data: pd.DataFrame,
    splits: DatasetSplits,
    config: AppConfig,
) -> TrainingResult:
    """Tune on validation, refit once, evaluate test once, then register."""

    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    splits.assert_disjoint()
    optuna_result = tune_with_optuna(splits, config)
    search_result = tune_with_random_search(splits, config)
    candidates = [optuna_result, search_result]
    selected = min(candidates, key=lambda candidate: candidate.validation_rmse)

    X_train_validation = pd.concat(
        [splits.X_train, splits.X_validation], axis=0, ignore_index=True
    )
    y_train_validation = pd.concat(
        [splits.y_train, splits.y_validation], axis=0, ignore_index=True
    )
    final_pipeline = build_model_pipeline(
        X_train_validation,
        reference_year=config.reference_year,
        random_state=config.random_state,
        model_params=selected.parameters,
    )
    final_pipeline.fit(X_train_validation, y_train_validation)

    # This is the only point in the workflow where test labels are accessed.
    test_predictions = final_pipeline.predict(splits.X_test)
    test_metrics = regression_metrics(splits.y_test, test_predictions)

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="train_tune_register") as run:
        mlflow.set_tags(
            {
                "stage": "tuned_candidate",
                "model_family": "catboost",
                "pipeline_input": "raw_real_estate_features",
                "split_policy": "disjoint_train_validation_test",
            }
        )
        mlflow.log_params(
            {
                "table_name": config.table_name,
                "target_column": config.target_column,
                "random_state": config.random_state,
                "validation_size": config.validation_size,
                "test_size": config.test_size,
                "reference_year": config.reference_year,
                "split_group_column": config.split_group_column or "none",
                "selected_method": selected.method,
                "dataset_sha256": dataset_fingerprint(data),
                **{f"model__{k}": v for k, v in selected.parameters.items()},
                **splits.summary(),
            }
        )
        mlflow.log_metrics(
            {
                "validation_rmse_selected": selected.validation_rmse,
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
            }
        )

        with TemporaryDirectory() as directory:
            reports = Path(directory)
            (reports / "candidates.json").write_text(
                json.dumps([asdict(candidate) for candidate in candidates], indent=2),
                encoding="utf-8",
            )
            (reports / "split_summary.json").write_text(
                json.dumps(splits.summary(), indent=2), encoding="utf-8"
            )
            mlflow.log_artifacts(str(reports), artifact_path="reports")

        input_example = prepare_model_contract_frame(splits.X_test.head(5))
        signature = infer_signature(input_example, final_pipeline.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=config.registered_model_name,
            serialization_format="cloudpickle",
            code_paths=[str(PROJECT_ROOT / "mlflow_project")],
            pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
        )

        client = MlflowClient()
        version = update_registered_version(
            client,
            model_name=config.registered_model_name,
            run_id=run.info.run_id,
            tags={
                "stage": "tuned_candidate",
                "rmse_test": f"{test_metrics['rmse']:.6f}",
                "r2_test": f"{test_metrics['r2']:.6f}",
                "raw_input_pipeline": "true",
            },
            description=(
                "Leak-free train/validation/test workflow. "
                f"Selected by validation RMSE with {selected.method}; "
                f"test RMSE={test_metrics['rmse']:.6f}, R2={test_metrics['r2']:.6f}. "
                f"Run: {run.info.run_id}"
            ),
            alias="candidate",
        )

    return TrainingResult(
        run_id=run.info.run_id,
        model_version=version,
        selected_method=selected.method,
        selected_parameters=selected.parameters,
        test_rmse=test_metrics["rmse"],
        test_mae=test_metrics["mae"],
        test_r2=test_metrics["r2"],
    )


def run_training(config: AppConfig) -> TrainingResult:
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
    return train_and_register(data, splits, config)


def main() -> None:
    result = run_training(AppConfig.from_env())
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
