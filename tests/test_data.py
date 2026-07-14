from __future__ import annotations

import pandas as pd
import pytest

from mlflow_project.data import parse_table_name, split_dataset, validate_dataset


def test_parse_table_name_accepts_valid_qualified_name() -> None:
    assert parse_table_name("public.real_estate") == ("public", "real_estate")


@pytest.mark.parametrize(
    "name",
    ["public.table;DROP TABLE users", "too.many.parts", "public.bad-name", ""],
)
def test_parse_table_name_rejects_sql_and_invalid_identifiers(name: str) -> None:
    with pytest.raises(ValueError):
        parse_table_name(name)


def test_split_is_deterministic_disjoint_and_complete(real_estate_frame: pd.DataFrame) -> None:
    first = split_dataset(real_estate_frame, validation_size=0.2, test_size=0.2)
    second = split_dataset(real_estate_frame, validation_size=0.2, test_size=0.2)

    first.assert_disjoint()
    summary = first.summary()
    assert summary["train_rows"] + summary["validation_rows"] + summary["test_rows"] == 120
    assert summary["train_groups"] > summary["validation_groups"] > 0
    assert summary["train_groups"] > summary["test_groups"] > 0
    assert first.train_positions.tolist() == second.train_positions.tolist()
    assert first.validation_positions.tolist() == second.validation_positions.tolist()
    assert first.test_positions.tolist() == second.test_positions.tolist()
    assert len(
        set(first.train_positions)
        | set(first.validation_positions)
        | set(first.test_positions)
    ) == len(real_estate_frame)

    groups = real_estate_frame["building_id"]
    train_groups = set(groups.iloc[first.train_positions])
    validation_groups = set(groups.iloc[first.validation_positions])
    test_groups = set(groups.iloc[first.test_positions])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_validate_dataset_rejects_missing_target(real_estate_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(real_estate_frame.drop(columns=["price"]))
