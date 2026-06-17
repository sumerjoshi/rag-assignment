import pytest
from pydantic import ValidationError

from src.models import AgentResponse, PDFEvidence, SQLEvidence


def test_evidence_source_defaults() -> None:
    # the literal source tag is what the union discriminates on
    assert SQLEvidence(query="SELECT 1", rows=[]).source == "sql"
    assert PDFEvidence(company_ticker="AAPL", chunk_id="c1", fiscal_year=2024, page_number=3, text="t").source == "pdf"


def test_agent_response_discriminates_evidence_union() -> None:
    # mixed evidence should parse into the right concrete types off the source tag
    resp = AgentResponse.model_validate(
        {
            "answer": "ok",
            "sources_used": "pdf",
            "evidence": [
                {"source": "sql", "query": "SELECT 1", "rows": []},
                {"source": "pdf", "company_ticker": "AAPL", "chunk_id": "c1", "fiscal_year": 2024, "page_number": 3, "text": "t", "score": 0.5},
            ],
        }
    )
    assert isinstance(resp.evidence[0], SQLEvidence)
    assert isinstance(resp.evidence[1], PDFEvidence)


def test_evidence_rejects_unknown_source() -> None:
    # an unrecognized discriminator should fail, not silently pass through
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(
            {"answer": "x", "sources_used": "pdf", "evidence": [{"source": "csv", "foo": 1}]}
        )


def test_sources_used_rejects_invalid_value() -> None:
    # "both" is not allowed by the current Literal["sql", "pdf"]
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({"answer": "x", "sources_used": "both", "evidence": []})
