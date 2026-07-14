from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def real_estate_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 120
    rooms = rng.integers(1, 5, size=rows)
    total_area = rng.uniform(25, 120, size=rows)
    floor = rng.integers(1, 20, size=rows)
    floors_total = np.maximum(floor, rng.integers(5, 30, size=rows))
    build_year = rng.integers(1940, 2025, size=rows)
    latitude = rng.normal(55.75, 0.08, size=rows)
    longitude = rng.normal(37.62, 0.1, size=rows)
    price = (
        total_area * 180_000
        + rooms * 300_000
        - (2026 - build_year) * 15_000
        + rng.normal(0, 300_000, size=rows)
    )

    return pd.DataFrame(
        {
            "flat_id": np.arange(rows),
            "building_id": rng.integers(1, 30, size=rows),
            "floor": floor,
            "kitchen_area": total_area * rng.uniform(0.12, 0.25, size=rows),
            "living_area": total_area * rng.uniform(0.45, 0.75, size=rows),
            "rooms": rooms,
            "is_apartment": rng.choice([True, False], size=rows),
            "studio": rng.choice([True, False], size=rows),
            "total_area": total_area,
            "price": price,
            "build_year": build_year,
            "building_type_int": rng.integers(0, 7, size=rows),
            "latitude": latitude,
            "longitude": longitude,
            "ceiling_height": rng.uniform(2.5, 3.2, size=rows),
            "flats_count": rng.integers(20, 800, size=rows),
            "floors_total": floors_total,
            "has_elevator": rng.choice([True, False], size=rows),
        }
    )
