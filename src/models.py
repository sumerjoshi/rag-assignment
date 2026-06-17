from typing import Annotated, Literal
from pydantic import BaseModel, Field


class ResultChunk(BaseModel):
    chunk_id: int 
    company_ticker: str 
    fiscal_year: int 
    page_number: int 
    section: str | None = None  # Optional Field
    company_name: str | None = None
    chunk_index: int | None = None # Optional Field
    text: str 
    score: int 

class SQLEvidence(BaseModel):
    source: Literal["sql"] = "sql"
    query: str 
    rows: list[dict]

class PDFEvidence(BaseModel):
    source: Literal["pdf"] = "pdf"
    company_ticker: str
    chunk_id: str 
    fiscal_year: int 
    page_number: int 
    section: str | None = None # Optional Field 
    text: str 
    score: float | None = None # Optional Field

Evidence = Annotated[SQLEvidence | PDFEvidence, Field(discriminator="source")]

class AgentResponse(BaseModel):
    answer: str 
    sources_used: Literal["sql", "pdf"]
    evidence: list[Evidence]

