"""Safe SQL extraction, dataset validation and leak-free splitting."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sqlalchemy import MetaData, Table, create_engine, select

from .config import DatabaseConfig

DATASET_COLUMNS = (
    "flat_id",
    "building_id",
    "floor",
    "kitchen_area",
    "living_area",
    "rooms",
    "is_apartment",
    "studio",
    "total_area",
    "price",
    "build_year",
    "building_type_int",
    "latitude",
    "longitude",
    "ceiling_height",
    "flats_count",
    "floors_total",
    "has_elevator",
)
MODEL_INPUT_COLUMNS = tuple(c for c in DATASET_COLUMNS if c != "price")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_table_name(qualified_name: str) -> tuple[str | None, str]:
    """Validate and split ``schema.table`` without treating it as SQL text."""

    parts = qualified_name.split(".")
    if len(parts) == 1:
        schema, table = None, parts[0]
    elif len(parts) == 2:
        schema, table = parts
    else:
        raise ValueError("table name must be 'table' or 'schema.table'")

    if not table or not _IDENTIFIER.fullmatch(table):
        raise ValueError(f"invalid table identifier: {table!r}")
    if schema is not None and not _IDENTIFIER.fullmatch(schema):
        raise ValueError(f"invalid schema identifier: {schema!r}")
    return schema, table


def load_dataset(
    database: DatabaseConfig,
    table_name: str,
    columns: Iterable[str] | None = None,
    target_column: str = "price",
) -> pd.DataFrame:
    """Load explicit columns through SQLAlchemy Core and close all resources."""

    requested = (
        tuple(c for c in DATASET_COLUMNS if c != "price") + (target_column,)
        if columns is None
        else tuple(columns)
    )
    if not requested:
        raise ValueError("at least one column must be requested")
    for column in requested:
        if not _IDENTIFIER.fullmatch(column):
            raise ValueError(f"invalid column identifier: {column!r}")

    schema, table = parse_table_name(table_name)
    engine = create_engine(database.url, pool_pre_ping=True)
    try:
        metadata = MetaData()
        reflected = Table(table, metadata, schema=schema, autoload_with=engine)
        missing = sorted(set(requested) - set(reflected.c.keys()))
        if missing:
            raise RuntimeError(
                f"Table {table_name} is missing required columns: {', '.join(missing)}"
            )
        statement = select(*(reflected.c[name] for name in requested))
        with engine.connect() as connection:
            data = pd.read_sql(statement, connection)
    finally:
        engine.dispose()

    validate_dataset(data, requested, target_column)
    return data


def validate_dataset(
    data: pd.DataFrame,
    required_columns: Iterable[str] = DATASET_COLUMNS,
    target_column: str = "price",
) -> None:
    if data.empty:
        raise ValueError("dataset is empty")
    if not data.columns.is_unique:
        raise ValueError("dataset contains duplicate column names")

    required = set(required_columns)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(missing)}")
    if target_column not in data:
        raise ValueError(f"target column {target_column!r} is missing")
    if data[target_column].isna().any():
        raise ValueError("target contains missing values")
    if not np.isfinite(pd.to_numeric(data[target_column], errors="coerce")).all():
        raise ValueError("target contains non-numeric or infinite values")


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_positions: np.ndarray
    validation_positions: np.ndarray
    test_positions: np.ndarray
    train_groups: np.ndarray | None = None
    validation_groups: np.ndarray | None = None
    test_groups: np.ndarray | None = None

    def assert_disjoint(self) -> None:
        train = set(map(int, self.train_positions))
        validation = set(map(int, self.validation_positions))
        test = set(map(int, self.test_positions))
        if train & validation or train & test or validation & test:
            raise AssertionError("train, validation and test positions must be disjoint")
        if len(train | validation | test) != (
            len(train) + len(validation) + len(test)
        ):
            raise AssertionError("split positions contain duplicates")

        if self.train_groups is not None:
            train_group_set = set(self.train_groups.tolist())
            validation_group_set = set(self.validation_groups.tolist())
            test_group_set = set(self.test_groups.tolist())
            if (
                train_group_set & validation_group_set
                or train_group_set & test_group_set
                or validation_group_set & test_group_set
            ):
                raise AssertionError(
                    "train, validation and test groups must be disjoint"
                )

    def summary(self) -> dict[str, int]:
        summary = {
            "train_rows": len(self.X_train),
            "validation_rows": len(self.X_validation),
            "test_rows": len(self.X_test),
        }
        if self.train_groups is not None:
            summary.update(
                {
                    "train_groups": len(set(self.train_groups.tolist())),
                    "validation_groups": len(
                        set(self.validation_groups.tolist())
                    ),
                    "test_groups": len(set(self.test_groups.tolist())),
                }
            )
        return summary


def split_dataset(
    data: pd.DataFrame,
    target_column: str = "price",
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    group_column: str | None = "building_id",
) -> DatasetSplits:
    """Create deterministic row- and group-disjoint train/validation/test splits."""

    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    positions = np.arange(len(data))
    relative_validation_size = validation_size / (1 - test_size)

    if group_column is not None:
        if group_column not in data:
            raise ValueError(f"group column {group_column!r} is missing")
        if data[group_column].isna().any():
            raise ValueError(f"group column {group_column!r} contains missing values")
        if data[group_column].nunique() < 3:
            raise ValueError("at least three distinct groups are required")

        groups = data[group_column].reset_index(drop=True)
        outer_split = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_validation_local, test_local = next(
            outer_split.split(positions, groups=groups)
        )
        train_validation_positions = positions[train_validation_local]
        test_positions = positions[test_local]

        inner_split = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_validation_size,
            random_state=random_state,
        )
        train_local, validation_local = next(
            inner_split.split(
                train_validation_positions,
                groups=groups.iloc[train_validation_positions],
            )
        )
        train_positions = train_validation_positions[train_local]
        validation_positions = train_validation_positions[validation_local]
        train_groups = groups.iloc[train_positions].to_numpy()
        validation_groups = groups.iloc[validation_positions].to_numpy()
        test_groups = groups.iloc[test_positions].to_numpy()
    else:
        train_validation_positions, test_positions = train_test_split(
            positions,
            test_size=test_size,
            random_state=random_state,
        )
        train_positions, validation_positions = train_test_split(
            train_validation_positions,
            test_size=relative_validation_size,
            random_state=random_state,
        )
        train_groups = validation_groups = test_groups = None

    features = data.drop(columns=[target_column])
    target = pd.to_numeric(data[target_column], errors="raise").astype(float)

    def take(frame: pd.DataFrame | pd.Series, idx: np.ndarray):
        return frame.iloc[idx].reset_index(drop=True)

    result = DatasetSplits(
        X_train=take(features, train_positions),
        X_validation=take(features, validation_positions),
        X_test=take(features, test_positions),
        y_train=take(target, train_positions),
        y_validation=take(target, validation_positions),
        y_test=take(target, test_positions),
        train_positions=train_positions,
        validation_positions=validation_positions,
        test_positions=test_positions,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )
    result.assert_disjoint()
    return result


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Return a stable content fingerprint for experiment lineage."""

    hashed = pd.util.hash_pandas_object(data, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()
