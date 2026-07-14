from __future__ import annotations

import pandas as pd

from mlflow_project.data import split_dataset
from mlflow_project.training import make_predefined_search_data


def test_search_uses_train_only_for_fit_and_validation_only_for_score(
    real_estate_frame: pd.DataFrame,
) -> None:
    splits = split_dataset(real_estate_frame)
    X_search, y_search, predefined = make_predefined_search_data(splits)
    train_indices, validation_indices = next(predefined.split())

    assert len(X_search) == len(splits.X_train) + len(splits.X_validation)
    assert len(y_search) == len(X_search)
    assert len(train_indices) == len(splits.X_train)
    assert len(validation_indices) == len(splits.X_validation)
    assert set(train_indices).isdisjoint(set(validation_indices))
