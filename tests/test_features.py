from __future__ import annotations

import numpy as np
import pandas as pd

from mlflow_project.data import split_dataset
from mlflow_project.features import (
    RealEstateFeatureBuilder,
    build_model_pipeline,
    prepare_model_contract_frame,
)


def test_feature_builder_removes_ids_and_adds_domain_features(
    real_estate_frame: pd.DataFrame,
) -> None:
    transformed = RealEstateFeatureBuilder(reference_year=2026).fit_transform(
        real_estate_frame.drop(columns=["price"])
    )

    assert "flat_id" not in transformed
    assert "building_id" not in transformed
    assert {
        "area_per_room",
        "living_area_share",
        "kitchen_area_share",
        "floor_ratio",
        "building_age",
        "flats_per_floor",
    } <= set(transformed.columns)
    assert np.isfinite(transformed["area_per_room"]).all()


def test_serializable_pipeline_accepts_raw_features(real_estate_frame: pd.DataFrame) -> None:
    splits = split_dataset(real_estate_frame)
    pipeline = build_model_pipeline(
        splits.X_train,
        reference_year=2026,
        random_state=42,
        model_params={"iterations": 10, "depth": 4, "feature_percentile": 75},
    )

    pipeline.fit(splits.X_train, splits.y_train)
    predictions = pipeline.predict(splits.X_validation)

    assert predictions.shape == (len(splits.X_validation),)
    assert np.isfinite(predictions).all()


def test_serving_contract_drops_ids_and_allows_nullable_numbers(
    real_estate_frame: pd.DataFrame,
) -> None:
    contract = prepare_model_contract_frame(real_estate_frame.drop(columns=["price"]))

    assert "flat_id" not in contract
    assert "building_id" not in contract
    assert not contract.select_dtypes(include=["integer"]).columns.tolist()
    assert contract["rooms"].dtype == "float64"
    assert contract["has_elevator"].dtype == "bool"
