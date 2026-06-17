import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import src.config as config
from src.config import ABSOLUTE_DB_PATH
from src.retrieval.sql import (
    CONTEXT_SCHEMA_STR,
    TABLES,
    build_sql_query_engine,
)


def test_context_string_has_the_important_rules() -> None:
    # fail if someone trims the gotchas the LLM needs
    assert "FY" in CONTEXT_SCHEMA_STR
    assert "dollars" in CONTEXT_SCHEMA_STR.lower()
    for table in TABLES:
        assert table in CONTEXT_SCHEMA_STR


def test_build_sql_query_engine_constructs() -> None:
    # smoke test, also needs configure_settings() first or it falls back to OpenAI
    config.configure_settings()
    engine = build_sql_query_engine()
    assert engine is not None


def test_engine_connection_is_read_only() -> None:
    # same read-only string the module uses, writes must be rejected
    engine = create_engine(f"sqlite:///file:{ABSOLUTE_DB_PATH}?mode=ro&uri=true")
    with pytest.raises(OperationalError):
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE should_not_work (id INTEGER)"))
