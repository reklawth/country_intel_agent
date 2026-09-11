'''
agent.py -- Build the country-intel agent using LangChain 1.x and a local Mixtral Large Language Model (LLM) served by vLLM.

LangChain 1.x is used to build the agent, which uses a local Mixtral LLM served by vLLM for natural language processing. 
The agent is configured to use the ChatOpenAI class from LangChain to connect to the Mixtral LLM via the vLLM API. Langchain
1.0 replaced the legacy AgentExecutor / create-*_agent constructors with with a single `create_agent` factory built on LangGraph.
It uses the model's NATIVE tool colling, so for Mixtral, you must serve vLLM with the mistral tool parser + a mistral tool-cool chat
template. The agent is designed to handle country intelligence queries and can be extended with additional tools and capabilities as needed.
'''

import json
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from llm import build_llm
from tools import ALL_TOOLS

# Names of the real tools, used to validate any tool call we recover from text.
_TOOL_NAMES = {t.name for t in ALL_TOOLS}


def _extract_text_tool_calls(text: str) -> list[dict]:
    """Recover tool calls that the model wrote as JSON text instead of emitting natively.

    Small local models (e.g. Mistral-7B) often narrate a call like
    `[{"name": "get_world_bank_indicator", "arguments": {...}}]` in the message body
    rather than using vLLM's native tool-call format, so the parser never runs them and
    placeholders like result["population"] leak into the answer. We scan the text for JSON
    objects/arrays and promote any whose "name" matches a real tool.
    """
    calls, decoder, i = [], json.JSONDecoder(), 0
    while i < len(text):
        if text[i] in "[{":
            try:
                obj, end = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                i += 1
                continue
            for item in (obj if isinstance(obj, list) else [obj]):
                if isinstance(item, dict) and item.get("name") in _TOOL_NAMES:
                    calls.append({
                        "name": item["name"],
                        "args": item.get("arguments") or item.get("args") or {},
                        "id": uuid.uuid4().hex[:9],
                        "type": "tool_call",
                    })
            i += end
        else:
            i += 1
    return calls


class ReliableToolCallMiddleware(AgentMiddleware):
    """Make native tool calling reliable with weak local models.

    (1) Forces tool_choice="required" on the first model call so the agent opens with a real
        tool call instead of a prose "here is how I would do it" plan.
    (2) After every model call, if the model produced no native tool calls but wrote one as
        text, promotes that text into a real tool call so it actually executes.

    Note: forcing tool calls on EVERY step (until "no progress") was tried and abandoned -- with
    several tools available a forced retry can always find some new tool to call, so it never
    stops, over-calling irrelevant tools and timing out. This model satisfices on multi-part
    questions; robust multi-tool chaining really needs a stronger tool-calling model.
    """

    def wrap_model_call(self, request, handler):
        # (1) No tool has run yet -> force the first turn to be a tool call.
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            request = request.override(tool_choice="required")

        response = handler(request)

        # (2) Recover any tool call the model wrote as text.
        for idx, message in enumerate(response.result):
            if isinstance(message, AIMessage) and not message.tool_calls:
                recovered = _extract_text_tool_calls(message.content or "")
                if recovered:
                    response.result[idx] = AIMessage(content="", tool_calls=recovered)
        return response

# System prompt (was the "ROLE" in LangChain 0.x) for the agent. This prompt is used to instruct the agent on how to behave and what its purpose is.
SYSTEM_PROMPT = """
You are a country-intel assistant that builds concise, factual country profiles from live tool data, in the spirit of a world factbook.
Always CALL the tools to get data -- never answer from memory, and never write a tool call as text.
Rules:
(1) Call get_country_profile FIRST to obtain a country's ISO3 code before calling the World Bank tool.
(2) Always state the YEAR for any statistic, as sources may differ.
(3) If a tool returns an 'error', try another approach -- NEVER invent data.
(4) Keep final answers tight and well organized.
"""

def build_agent(verbose: bool = True):
    ''' Return a complied LangGraph agent.
    
    Invoke it with a messages list, e.g.:
        agent.invoke({"messages": [{"role": "user", "content": "..."}]})
    and read the final answer from result["messages"][-1].content.

    `verbose=True` sets debug mode, which prints the graph's step-by-step execution -- the 1.x stand-in for the old AgentExecutor(verbose=True) trace.
    '''
    return create_agent(
        model=build_llm(),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[ReliableToolCallMiddleware()],
        debug=verbose,
    )