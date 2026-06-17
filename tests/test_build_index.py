from pathlib import Path

import pytest
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from src.ingest.build_index import IndexBuilder

VALID_TICKERS = {"AAPL", "MSFT", "GOOGL"}


def test_parse_metadata_pulls_ticker_and_year() -> None:
    meta = IndexBuilder()._parse_metadata("/some/dir/AAPL_FY2024_10-K.pdf")
    assert meta == {"company_ticker": "AAPL", "fiscal_year": 2024}
    # year needs to be an int for the pydantic models later
    assert isinstance(meta["fiscal_year"], int)


def test_parse_metadata_rejects_bad_filename() -> None:
    with pytest.raises(ValueError):
        IndexBuilder()._parse_metadata("not_a_filing.pdf")


def test_get_pdf_file_paths_returns_six_real_files() -> None:
    paths = IndexBuilder()._get_pdf_file_paths()
    assert len(paths) == 6
    assert all(Path(p).exists() for p in paths)


def test_load_documents_attaches_metadata() -> None:
    docs = IndexBuilder().load_documents()
    assert len(docs) > 0
    for d in docs:
        assert set(d.metadata) >= {"company_ticker", "fiscal_year", "page_number"}
        assert d.metadata["company_ticker"] in VALID_TICKERS
        # pages start at 1, not 0
        assert d.metadata["page_number"] >= 1
        # we skip blank pages so text should never be empty
        assert d.text.strip()


def test_build_index_builds_nodes_without_network() -> None:
    # mock embeddings so we don't hit fireworks in a unit test
    Settings.embed_model = MockEmbedding(embed_dim=8)
    docs = [
        Document(text="Apple revenue grew.", metadata={"company_ticker": "AAPL", "fiscal_year": 2024, "page_number": 1}),
        Document(text="Microsoft cloud expanded.", metadata={"company_ticker": "MSFT", "fiscal_year": 2024, "page_number": 1}),
    ]
    index = IndexBuilder().build_index(docs)
    assert isinstance(index, VectorStoreIndex)
    assert len(index.docstore.docs) > 0
