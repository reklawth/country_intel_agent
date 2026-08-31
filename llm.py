'''
llm.py -- Connect LangChain to a local Mixtral Large Language Model (LLM) served by vLLM and using the LLM API.

vLLM eposes a REST API endpoint that is compatible with the OpenAI API, allowing you to use LangChain with Mixtral LLMs.
Therefore we can use ChatOpenAI from LangChain to connect to the Mixtral LLM served by vLLM..  The api_key is required by the client library, 
but it is not used for authentication with vLLM. You can set it to any value.

Configure via environment variables:
- `VLLM_HOST`: The host of the vLLM server (default: "http://localhost:8000/v1")
- `VLLM_MODEL`: The name of the Mixtral model (default: "mixtral")
- `VLLM_API_KEY`: The API key for the vLLM server (default: "test")
'''

import os
from langchain.chat_models import ChatOpenAI

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
        vllm_model=os.getenv("VLLM_MODEL", "mixtral"),
        vllm_host=os.getenv("VLLM_HOST", "http://localhost:8000/v1"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "test"),
        temperature=temperature,
        timeout=60.0,
        max_retries=3,
    )