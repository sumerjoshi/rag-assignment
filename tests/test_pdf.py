from llama_index.core.schema import NodeWithScore, TextNode

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
