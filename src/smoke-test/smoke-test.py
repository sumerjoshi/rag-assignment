#!/usr/bin/env python

import os
import asyncio
from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult, AgentStream, AgentOutput
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

load_dotenv()

def query_financials(ticker: str, year: int) -> str:
    return f"{ticker} FY{year} total revenue: $415.2B" #stub

def search_filings(query: str) -> str:
    return(
        "[10-K excerpt — Item 1A Risk Factors, p. 12] The company relies on "
        "third-party manufacturers and a limited number of component suppliers, "
        "some of which are single-source. Supply chain disruptions — from "
        "geopolitical tensions, natural disasters, public health emergencies, or "
        "manufacturing capacity constraints — could impair its ability to obtain "
        "components, increase costs, and delay product availability. A significant "
        "portion of manufacturing is concentrated in specific geographic regions, "
        "which heightens exposure to regional disruption."
    )

tools = [
    FunctionTool.from_defaults(query_financials,
                               name="query_financials",
                               description="Look up structured numerical financial for a company and fiscal year"),
    FunctionTool.from_defaults(search_filings,
                               name="search_filings",
                               description="Search the narrative text of the 10K for qualitative information. Use for business strategy but not for raw numbers - use query financials for that")
]

llm = OpenAILike(
        model=os.getenv("FIREWORKS_LLM_MODEL"),
        api_key=os.getenv("FIREWORKS_API_KEY"),
        api_base=os.getenv("FIREWORKS_BASE_URL"),
        is_chat_model=True,
        is_function_calling_model=True
)

# seeing if this setup works fireworks with something like this. testing keys here basically.
# trying OpenAILike python package.
print(llm.metadata.is_function_calling_model)
print(llm.complete("Say hello in 5 words"))

agent = FunctionAgent(
    llm=llm,
    tools=tools,
    timeout=120,
    verbose=True
)

async def run_code():
    # running async code here
    handler = agent.run(user_msg="What was Apple's revenue?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print("TOOL CALL", ev.tool_name, ev.tool_kwargs)
    result = await handler
    print("QF:", result)

    handler_sf = agent.run(user_msg="What supply chain risks were disclosed in Google's 10K", max_iterations=10)
    async for ev in handler_sf.stream_events():
        if isinstance(ev, ToolCall):
            print("TOOL CALL", ev.tool_name, ev.tool_kwargs)
    result_two = await handler_sf
    print("SF:", result_two)

if __name__ == "__main__":
    asyncio.run(run_code())
