"""Helpers for updating exactly the model version created by one MLflow run."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def find_version_for_run(versions: Iterable[Any], run_id: str) -> Any:
    matches = [version for version in versions if getattr(version, "run_id", None) == run_id]
    if not matches:
        raise RuntimeError(f"No registered model version found for run_id={run_id}")
    return max(matches, key=lambda version: int(version.version))


def update_registered_version(
    client: Any,
    *,
    model_name: str,
    run_id: str,
    tags: dict[str, str],
    description: str,
    alias: str | None = "candidate",
) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    created = find_version_for_run(versions, run_id)
    version = str(created.version)

    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, str(key), str(value))
    client.update_model_version(
        name=model_name,
        version=version,
        description=description,
    )
    if alias and hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(model_name, alias, version)
    return version
