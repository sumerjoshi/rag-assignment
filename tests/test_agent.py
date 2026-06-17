import asyncio
from typing import Any, cast

import pytest
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.tools import FunctionTool

import src.agent.agent as agent_module
from src.models import Evidence, PDFEvidence, SQLEvidence


class _FakeResp:
    def __init__(self, metadata: dict[str, Any] | None) -> None:
        self.metadata = metadata

    def __str__(self) -> str:
        return "tool said this"


class _FakeEngine:
    def __init__(self, metadata: dict[str, Any] | None) -> None:
        self._metadata = metadata

    def query(self, question: str) -> _FakeResp:
        return _FakeResp(self._metadata)


def _tool(tools: list[Any], name: str) -> FunctionTool:
    return next(cast(FunctionTool, t) for t in tools if cast(FunctionTool, t).metadata.name == name)


def test_make_tools_returns_named_tools() -> None:
    names = {cast(FunctionTool, t).metadata.name for t in agent_module._make_tools([])}
    assert names == {"query_financials", "search_filings"}


def test_sql_tool_collects_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {"sql_query": "SELECT net_income FROM income_statements", "result": [(101832000000,)], "col_keys": ["net_income"]}
    monkeypatch.setattr(agent_module, "build_sql_query_engine", lambda: _FakeEngine(meta))
    evidence: list[Evidence] = []
    _tool(agent_module._make_tools(evidence), "query_financials").call(question="msft net income 2025")
    assert len(evidence) == 1
    assert isinstance(evidence[0], SQLEvidence)
    assert evidence[0].rows == [{"net_income": 101832000000}]


def test_pdf_tool_collects_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    item = PDFEvidence(company_ticker="AAPL", chunk_id="c1", fiscal_year=2024, page_number=3, text="risk", score=0.5)
    monkeypatch.setattr(agent_module, "query_pdf", lambda q: ("apple risks", [item]))
    evidence: list[Evidence] = []
    _tool(agent_module._make_tools(evidence), "search_filings").call(question="apple risks")
    assert evidence == [item]


def test_sql_tool_handles_empty_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    # no metadata should not crash, just produce empty evidence
    monkeypatch.setattr(agent_module, "build_sql_query_engine", lambda: _FakeEngine(None))
    evidence: list[Evidence] = []
    _tool(agent_module._make_tools(evidence), "query_financials").call(question="x")
    assert isinstance(evidence[0], SQLEvidence)
    assert evidence[0].rows == []
    assert evidence[0].query == ""


def test_pdf_tool_no_hits_collects_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # a search that finds nothing should leave evidence empty
    monkeypatch.setattr(agent_module, "query_pdf", lambda q: ("no relevant filings", []))
    evidence: list[Evidence] = []
    _tool(agent_module._make_tools(evidence), "search_filings").call(question="unrelated")
    assert evidence == []


class _FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        pass

    async def run(self, user_msg: str) -> str:
        return "final synthesized answer"


def test_answer_dedupes_sources_and_assembles_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # mock out the agent run; seed evidence as if one sql + two pdf tools fired
    Settings.llm = MockLLM()

    def fake_make_tools(evidence: list[Evidence]) -> list[Any]:
        evidence.append(SQLEvidence(query="SELECT 1", rows=[{"x": 1}]))
        evidence.append(PDFEvidence(company_ticker="AAPL", chunk_id="a", fiscal_year=2024, page_number=1, text="t", score=0.1))
        evidence.append(PDFEvidence(company_ticker="AAPL", chunk_id="b", fiscal_year=2024, page_number=2, text="u", score=0.2))
        return []

    monkeypatch.setattr(agent_module, "configure_settings", lambda: None)
    monkeypatch.setattr(agent_module, "_make_tools", fake_make_tools)
    monkeypatch.setattr(agent_module, "FunctionAgent", _FakeAgent)

    resp = asyncio.run(agent_module.answer("any question"))
    assert resp.answer == "final synthesized answer"
    # three evidence items, but sources deduped + sorted
    assert len(resp.evidence) == 3
    assert resp.sources_used == ["pdf", "sql"]


def test_answer_with_no_tool_calls_has_empty_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    # if the agent answers without calling a tool, sources_used is empty, not an error
    Settings.llm = MockLLM()
    monkeypatch.setattr(agent_module, "configure_settings", lambda: None)
    monkeypatch.setattr(agent_module, "_make_tools", lambda evidence: [])
    monkeypatch.setattr(agent_module, "FunctionAgent", _FakeAgent)

    resp = asyncio.run(agent_module.answer("hello"))
    assert resp.sources_used == []
    assert resp.evidence == []
