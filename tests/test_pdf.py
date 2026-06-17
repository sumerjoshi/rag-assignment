import types

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

import src.retrieval.pdf as pdf_module
from src.models import PDFEvidence
from src.retrieval.pdf import convert_to_pdf_evidence


def _node(score: float | None, **meta: object) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text="risk factors text", metadata=meta, id_="chunk-xyz"), score=score)


def test_convert_maps_node_to_pdf_evidence() -> None:
    [ev] = convert_to_pdf_evidence([_node(0.73, company_ticker="AAPL", fiscal_year=2024, page_number=10)])
    assert isinstance(ev, PDFEvidence)
    assert ev.company_ticker == "AAPL"
    assert ev.fiscal_year == 2024
    assert ev.page_number == 10
    assert ev.chunk_id == "chunk-xyz"
    assert ev.text == "risk factors text"
    assert ev.score == 0.73
    assert ev.source == "pdf"


def test_convert_defaults_missing_score_to_zero() -> None:
    # score is Optional on the node, the mapping should coerce None -> 0.0
    [ev] = convert_to_pdf_evidence([_node(None, company_ticker="MSFT", fiscal_year=2025, page_number=1)])
    assert ev.score == 0.0


def test_convert_empty_returns_empty() -> None:
    # no retrieval hits should not blow up
    assert convert_to_pdf_evidence([]) == []


def _stub_loading(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    # replace the disk/network boundaries so _get_index runs without loading anything
    def fake_load(sc: object) -> object:
        events.append("load")
        return object()

    monkeypatch.setattr(pdf_module, "_index", None)
    monkeypatch.setattr(pdf_module, "ABSOLUTE_VECTOR_STORE_PATH", types.SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(pdf_module, "configure_settings", lambda: events.append("configure"))
    monkeypatch.setattr(pdf_module, "StorageContext", types.SimpleNamespace(from_defaults=lambda **kw: None))
    monkeypatch.setattr(pdf_module, "load_index_from_storage", fake_load)


def test_get_index_configures_settings_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    # regression for the bug: the existing-index path must set the embed model first
    events: list[str] = []
    _stub_loading(monkeypatch, events)
    pdf_module._get_index()
    assert events == ["configure", "load"]


def test_get_index_loads_once_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    # second call should reuse the cached index, not load again
    events: list[str] = []
    _stub_loading(monkeypatch, events)
    first = pdf_module._get_index()
    second = pdf_module._get_index()
    assert first is second
    assert events.count("load") == 1
