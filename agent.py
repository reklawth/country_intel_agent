'''
agent.py -- Build the country-intel agent using LangChain 1.x and a local Mixtral Large Language Model (LLM) served by vLLM.

LangChain 1.x is used to build the agent, which uses a local Mixtral LLM served by vLLM for natural language processing. 
The agent is configured to use the ChatOpenAI class from LangChain to connect to the Mixtral LLM via the vLLM API. Langchain
1.0 replaced the legacy AgentExecutor / create-*_agent constructors with with a single `create_agent` factory built on LangGraph.
It uses the model's NATIVE tool colling, so for Mixtral, you must serve vLLM with the mistral tool parser + a mistral tool-cool chat
template. The agent is designed to handle country intelligence queries and can be extended with additional tools and capabilities as needed.
'''

from langchain.agents import create_agent
from llm import build_llm
from tools import ALL_TOOLS

# System prompt (was the "ROLE" in LangChain 0.x) for the agent. This prompt is used to instruct the agent on how to behave and what its purpose is.
SYSTEM_PROMPT = """
You are a country-intel agent that provides information about countries. You have access to a set of tools that you can use to answer questions about countries.
You build, concise factual country profiles by combining data from several tools, in the spirit of a world factbook.
Rules:
(1) Always resolve a country with get_country_profile FIRST to obtain its ISO3 code before calling the World Bank tool.
(2) Always state the YEAR for any statistic as sources may differ.
(3) If a tool returns an 'error', adapt or try another approach -- NEVER invent data.
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
        model=build_llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        debug=verbose,
    )