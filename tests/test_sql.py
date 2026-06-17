import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import src.config as config
from src.config import ABSOLUTE_DB_PATH
from src.models import SQLEvidence
from src.retrieval.sql import (
    CONTEXT_SCHEMA_STR,
    TABLES,
    build_sql_query_engine,
    convert_to_sql_evidence,
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


def test_convert_to_sql_evidence_zips_rows_into_dicts() -> None:
    # shape comes from the real engine: result is tuples, col_keys are the columns
    meta = {
        "sql_query": "SELECT revenue FROM income_statements WHERE company_ticker = 'AAPL'",
        "result": [(416161000000,)],
        "col_keys": ["revenue"],
    }
    ev = convert_to_sql_evidence(meta)
    assert isinstance(ev, SQLEvidence)
    assert ev.source == "sql"
    assert ev.query == meta["sql_query"]
    assert ev.rows == [{"revenue": 416161000000}]


def test_convert_to_sql_evidence_handles_empty_result() -> None:
    # a query with no rows should give empty evidence, not crash
    ev = convert_to_sql_evidence({"sql_query": "SELECT 1 WHERE 0", "result": [], "col_keys": []})
    assert ev.rows == []


def test_convert_to_sql_evidence_falls_back_when_col_keys_missing() -> None:
    # missing col_keys must not produce empty dicts (would render as a columnless table)
    ev = convert_to_sql_evidence({"sql_query": "SELECT ...", "result": [(2025, 416161000000)], "col_keys": []})
    assert ev.rows == [{"col_0": 2025, "col_1": 416161000000}]
