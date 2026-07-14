"""Serializable raw-data feature engineering and model pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RealEstateFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build deterministic domain features from the raw real-estate schema."""

    def __init__(
        self,
        reference_year: int = 2026,
        drop_columns: tuple[str, ...] = ("flat_id", "building_id"),
    ) -> None:
        self.reference_year = reference_year
        self.drop_columns = drop_columns

    @staticmethod
    def _require_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RealEstateFeatureBuilder expects a pandas DataFrame")

    def fit(self, X: pd.DataFrame, y: Any = None) -> RealEstateFeatureBuilder:
        self._require_frame(X)
        if self.reference_year < 1900:
            raise ValueError("reference_year must be a four-digit calendar year")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        numerator_numeric = pd.to_numeric(numerator, errors="coerce")
        denominator_numeric = pd.to_numeric(denominator, errors="coerce")
        return numerator_numeric.div(denominator_numeric.replace(0, np.nan))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_frame(X)
        transformed = X.copy()
        transformed = transformed.drop(
            columns=[c for c in self.drop_columns if c in transformed],
            errors="ignore",
        )

        if "building_type_int" in transformed:
            transformed["building_type_int"] = (
                transformed["building_type_int"].astype("Int64").astype("string")
            )
        if {"total_area", "rooms"} <= set(transformed):
            transformed["area_per_room"] = self._safe_divide(
                transformed["total_area"], transformed["rooms"]
            )
        if {"living_area", "total_area"} <= set(transformed):
            transformed["living_area_share"] = self._safe_divide(
                transformed["living_area"], transformed["total_area"]
            )
        if {"kitchen_area", "total_area"} <= set(transformed):
            transformed["kitchen_area_share"] = self._safe_divide(
                transformed["kitchen_area"], transformed["total_area"]
            )
        if "floor" in transformed:
            floor = pd.to_numeric(transformed["floor"], errors="coerce")
            transformed["is_first_floor"] = floor.eq(1)
            if "floors_total" in transformed:
                floors_total = pd.to_numeric(
                    transformed["floors_total"], errors="coerce"
                )
                transformed["is_top_floor"] = floor.eq(floors_total)
                transformed["floor_ratio"] = self._safe_divide(
                    floor, floors_total
                )
        if "build_year" in transformed:
            build_year = pd.to_numeric(transformed["build_year"], errors="coerce")
            transformed["building_age"] = (self.reference_year - build_year).clip(
                lower=0, upper=300
            )
        if {"flats_count", "floors_total"} <= set(transformed):
            transformed["flats_per_floor"] = self._safe_divide(
                transformed["flats_count"], transformed["floors_total"]
            )

        return transformed.replace([np.inf, -np.inf], np.nan)


DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "iterations": 600,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
}


def prepare_model_contract_frame(
    frame: pd.DataFrame, *, drop_identifiers: bool = True
) -> pd.DataFrame:
    """Create a serving example that permits missing numeric values in MLflow."""

    prepared = frame.copy()
    if drop_identifiers:
        prepared = prepared.drop(
            columns=[c for c in ("flat_id", "building_id") if c in prepared],
            errors="ignore",
        )
    integer_columns = prepared.select_dtypes(include=["integer"]).columns
    for column in integer_columns:
        prepared[column] = prepared[column].astype("float64")
    return prepared


class CatBoostRegressorAdapter(RegressorMixin, BaseEstimator):
    """Expose CatBoost through sklearn's estimator protocol and tag system."""

    def __init__(
        self,
        iterations: int = 600,
        depth: int = 8,
        learning_rate: float = 0.08,
        l2_leaf_reg: float = 3.0,
        subsample: float = 1.0,
        colsample_bylevel: float = 1.0,
        loss_function: str = "RMSE",
        random_seed: int = 42,
        verbose: bool = False,
        allow_writing_files: bool = False,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files

    def fit(self, X: Any, y: Any) -> CatBoostRegressorAdapter:
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("CatBoostRegressorAdapter must be fitted before predict")
        return np.asarray(self.model_.predict(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("model has not been fitted")
        return np.asarray(self.model_.feature_importances_)


def build_model_pipeline(
    X_train: pd.DataFrame,
    *,
    reference_year: int,
    random_state: int,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build one serializable pipeline accepting the original raw feature frame."""

    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    percentile = int(params.pop("feature_percentile", 100))
    if not 1 <= percentile <= 100:
        raise ValueError("feature_percentile must be between 1 and 100")

    feature_builder = RealEstateFeatureBuilder(reference_year=reference_year)
    preview = feature_builder.fit_transform(X_train)
    categorical_columns = preview.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric_columns = preview.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        raise ValueError("no numeric features available after feature engineering")

    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = CatBoostRegressorAdapter(
        **params,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    return Pipeline(
        steps=[
            ("features", feature_builder),
            ("preprocess", preprocessor),
            ("select", SelectPercentile(score_func=f_regression, percentile=percentile)),
            ("model", model),
        ]
    )
