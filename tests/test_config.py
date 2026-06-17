import importlib
from pathlib import Path

import pytest

import src.config as config


def test_require_returns_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("SOME_TEST_VAR", "hello")
    assert config._require("SOME_TEST_VAR") == "hello"


def test_require_raises_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("SOME_TEST_VAR", raising=False)
    with pytest.raises(ValueError):
        config._require("SOME_TEST_VAR")


def test_require_raises_when_empty(monkeypatch) -> None:
    # empty should count as missing
    monkeypatch.setenv("SOME_TEST_VAR", "")
    with pytest.raises(ValueError):
        config._require("SOME_TEST_VAR")


def test_data_paths_are_absolute_and_exist() -> None:
    # relative .env values should resolve to real absolute paths
    assert config.ABSOLUTE_DB_PATH.is_absolute()
    assert config.ABSOLUTE_PDF_DIR_PATH.is_absolute()
    assert config.ABSOLUTE_DB_PATH.exists()
    assert config.ABSOLUTE_PDF_DIR_PATH.exists()


def test_db_path_lands_under_repo_root() -> None:
    # guards the old parents[2] bug that pointed outside the repo
    repo_root = Path(__file__).resolve().parents[1]
    assert str(config.ABSOLUTE_DB_PATH).startswith(str(repo_root))


def test_absolute_env_path_overrides(monkeypatch, tmp_path) -> None:
    # an absolute DATABASE_PATH should work
    custom = tmp_path / "custom.db"
    monkeypatch.setenv("DATABASE_PATH", str(custom))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.ABSOLUTE_DB_PATH == custom
    finally:
        # reset module state for the other tests
        monkeypatch.delenv("DATABASE_PATH", raising=False)
        importlib.reload(config)
