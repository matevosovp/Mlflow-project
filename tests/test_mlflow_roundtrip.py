from __future__ import annotations

import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from mlflow_project.config import PROJECT_ROOT
from mlflow_project.data import split_dataset
from mlflow_project.features import build_model_pipeline, prepare_model_contract_frame


def test_pipeline_round_trip_through_mlflow(
    tmp_path, real_estate_frame: pd.DataFrame
) -> None:
    splits = split_dataset(real_estate_frame)
    pipeline = build_model_pipeline(
        splits.X_train,
        reference_year=2026,
        random_state=42,
        model_params={"iterations": 10, "depth": 4, "feature_percentile": 75},
    )
    pipeline.fit(splits.X_train, splits.y_train)
    input_example = prepare_model_contract_frame(splits.X_validation.head(5))
    expected = pipeline.predict(input_example)
    model_path = tmp_path / "model"

    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=model_path,
        signature=infer_signature(input_example, expected),
        input_example=input_example,
        serialization_format="cloudpickle",
        code_paths=[str(PROJECT_ROOT / "mlflow_project")],
        pip_requirements=str(PROJECT_ROOT / "requirements.txt"),
    )
    assert (model_path / "code" / "mlflow_project" / "features.py").is_file()
    loaded = mlflow.sklearn.load_model(model_path)

    np.testing.assert_allclose(loaded.predict(input_example), expected)
