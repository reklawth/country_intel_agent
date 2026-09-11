'''
llm.py -- Connect LangChain to a local Mixtral Large Language Model (LLM) served by vLLM and using the LLM API.

vLLM eposes a REST API endpoint that is compatible with the OpenAI API, allowing you to use LangChain with Mixtral LLMs.
Therefore we can use ChatOpenAI from LangChain to connect to the Mixtral LLM served by vLLM..  The api_key is required by the client library, 
but it is not used for authentication with vLLM. You can set it to any value.

Configure via environment variables:
- `VLLM_HOST`: The host of the vLLM server (default: "http://localhost:8000/v1")
- `VLLM_MODEL`: The name of the served model (default: "mistralai/Mistral-7B-Instruct-v0.3", which supports native tool calling)
- `VLLM_API_KEY`: The API key for the vLLM server (default: "test")
- `VLLM_TIMEOUT`: Per-request timeout in seconds (default: 180)
'''

import os
from langchain_openai import ChatOpenAI

def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance that connects to a local Mixtral LLM served by vLLM.

    Args:
        temperature (float): The temperature for the LLM. Default is 0.0.
        This produces reproducible, deterministic output. Higher values produce more random output.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI configured to connect to the Mixtral LLM.
    """

    return ChatOpenAI(
        model=os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
        base_url=os.getenv("VLLM_HOST", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "test"),
        temperature=temperature,
        # Heavy multi-tool tasks (e.g. multi-country comparisons) make several calls and a long
        # synthesis; a single generation on a local 7B can exceed 60s. Default to 180s, tunable
        # via VLLM_TIMEOUT.
        timeout=float(os.getenv("VLLM_TIMEOUT", "180")),
        max_retries=3,
    )