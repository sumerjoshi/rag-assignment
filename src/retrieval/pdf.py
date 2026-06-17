
from typing import cast
from src.config import ABSOLUTE_VECTOR_STORE_PATH, configure_settings
from llama_index.core import VectorStoreIndex
from src.ingest.build_index import IndexBuilder
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.vector_stores import MetadataFilters
from src.models import PDFEvidence
from llama_index.core.schema import NodeWithScore

# Make it so that you only load the index once
# so you are not loading it every single time.
_index: VectorStoreIndex | None = None

def _get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        # loading (not just querying) needs Settings.embed_model, or it falls back to
        # OpenAI. the build path sets this, but the existing-index path must too.
        configure_settings()
        if not ABSOLUTE_VECTOR_STORE_PATH.exists():
            IndexBuilder().run()
        sc = StorageContext.from_defaults(persist_dir=str(ABSOLUTE_VECTOR_STORE_PATH))
        # load_index_from_storage returns a generic BaseIndex; have to cast it
        _index = cast(VectorStoreIndex, load_index_from_storage(sc))
    return _index
        
# build the pdf engine getting top_k here with filters
def build_pdf_query_engine(top_k: int = 5, filters: MetadataFilters | None = None) -> BaseQueryEngine:
    return _get_index().as_query_engine(similarity_top_k=top_k, filters=filters)


# convert to the pdf evidence object using Node metadata
def convert_to_pdf_evidence(source_nodes: list[NodeWithScore]) -> list[PDFEvidence]:
    evidence: list[PDFEvidence] = []
    for n in source_nodes:
        # TODO: this assumes every node has the expected metadata keys. ingestion
        # guarantees it today, but a malformed node would raise a cryptic KeyError.
        # harden with meta.get(...) + skip/validate if we ever ingest other sources.
        meta = n.node.metadata
        evidence.append(
            PDFEvidence(
                company_ticker=meta["company_ticker"],
                fiscal_year=meta["fiscal_year"],
                page_number=meta["page_number"],
                chunk_id=n.node.node_id,
                text=n.node.get_content(),
                score=n.score or 0.0
            )
        )
    return evidence

# Query the PDF with a questions and return the string response and the pdf evidence
def query_pdf(question: str, top_k: int = 5, filters: MetadataFilters | None = None) -> tuple[str, list[PDFEvidence]]:
    response = build_pdf_query_engine(top_k=top_k, filters=filters).query(question)
    return str(response), convert_to_pdf_evidence(response.source_nodes)
