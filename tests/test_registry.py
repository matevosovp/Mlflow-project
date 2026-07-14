from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlflow_project.registry import find_version_for_run, update_registered_version


@dataclass
class Version:
    version: str
    run_id: str


class FakeClient:
    def __init__(self) -> None:
        self.versions = [Version("11", "other"), Version("9", "current")]
        self.tagged: list[tuple[str, str, str, str]] = []
        self.alias: tuple[str, str, str] | None = None

    def search_model_versions(self, _query: str):
        return self.versions

    def set_model_version_tag(self, name: str, version: str, key: str, value: str):
        self.tagged.append((name, version, key, value))

    def update_model_version(self, **_kwargs):
        return None

    def set_registered_model_alias(self, name: str, alias: str, version: str):
        self.alias = (name, alias, version)


def test_find_version_filters_by_run_id_not_latest_version() -> None:
    versions = [Version("12", "other"), Version("4", "mine"), Version("7", "mine")]
    assert find_version_for_run(versions, "mine").version == "7"


def test_find_version_fails_when_registration_is_missing() -> None:
    with pytest.raises(RuntimeError, match="No registered model version"):
        find_version_for_run([Version("1", "other")], "mine")


def test_update_tags_only_current_run_version() -> None:
    client = FakeClient()
    version = update_registered_version(
        client,
        model_name="price_model",
        run_id="current",
        tags={"stage": "candidate"},
        description="verified",
    )

    assert version == "9"
    assert client.tagged == [("price_model", "9", "stage", "candidate")]
    assert client.alias == ("price_model", "candidate", "9")
