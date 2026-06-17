
from typing import Any, Callable, Literal, cast
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool, BaseTool
from llama_index.core import Settings
from src.config import configure_settings
from src.models import AgentResponse, Evidence
from src.retrieval.sql import build_sql_query_engine, convert_to_sql_evidence
from src.retrieval.pdf import query_pdf

# the system prompt is the routing logic: which tool to use for which kind of question
ROUTING_PROMPT = (
    "You are a financial analyst assistant answering questions about AAPL (Apple), "
    "MSFT (Microsoft), and GOOGL (Alphabet) based on their 10-K filings for fiscal "
    "years 2023-2025.\n"
    "You have two tools:\n"
    "- query_financials: exact numeric and structured data (revenue, net income, "
    "margins, segment and geographic revenue, balance sheet items). Always use this "
    "for figures.\n"
    "- search_filings: narrative and qualitative content (risk factors, strategy, "
    "MD&A, business description). Do not use this for exact numbers.\n"
    "Call both tools when a question needs numbers and narrative together.\n"
    "When a tool returns a figure or rows, report that value directly. Do not second-guess "
    "whether the data should exist or reason about fiscal calendars; the tools are "
    "authoritative. Ground every answer in the tool outputs, never invent figures, and only "
    "say data is unavailable if a tool actually returns no rows."
)

# just like smoke-test.py, we are now combining the tools here
# in one function so we can pick either or with evidence or a response
def _make_tools(evidence: list[Evidence]) -> list[BaseTool | Callable[..., Any]]:
    def query_financials(question: str) -> str:
        """Numeric financials that are structured: revenue, net income, segments, balance sheet."""
        resp = build_sql_query_engine().query(question)
        # resp.metadata holds the executed sql + rows; convert handles tuple->dict + None
        evidence.append(convert_to_sql_evidence(resp.metadata or {}))
        return str(resp)

    def search_filings(question: str) -> str:
        """Narrative 10-k test: risk factors, strategy, MD&A. Not for exact numbers."""
        answer, pdf_ev = query_pdf(question)
        evidence.extend(pdf_ev)
        return answer
    
    return [
        FunctionTool.from_defaults(query_financials),
        FunctionTool.from_defaults(search_filings),
    ]

# answer method that basically routes to the 
# right answer depending question parameters
async def answer(question: str) -> AgentResponse:
    configure_settings()
    evidence: list[Evidence] = []
    agent = FunctionAgent(
        llm=Settings.llm, tools=_make_tools(evidence=evidence),
        system_prompt=ROUTING_PROMPT)
    result = await agent.run(user_msg=question)
    # set comprehension over the union yields str; cast back to the literal the model expects
    sources = cast(list[Literal["sql", "pdf"]], sorted({e.source for e in evidence}))
    return AgentResponse(answer=str(result), sources_used=sources, evidence=evidence)