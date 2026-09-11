"""
main.py -- Run the country-intel agent using LangChain 1.x and a local Mixtral Large Language Model (LLM) served by vLLM.

Usage:
    python main.py                          # run the built-in demo tasks
    python main.py "your own question"      # ask the agent a question about a country
    VERBOSE-0 python main.py                # hide the step-by-step execution trace (default is VERBOSE-1)

Prerequisites:
    A local Mixtral LLM served by vLLM must be running and accessible. The agent uses the ChatOpenAI class from LangChain to connect to the Mixtral LLM via the vLLM API.
    (see README -- create_agent relies on the model's NATIVE tool colling, so for Mixtral, you must serve vLLM with the mistral tool parser + a mistral tool-cool chat template.)
"""

import os
import sys

from dotenv import load_dotenv

# Load .env before anything reads os.getenv (llm.py and tools.py read their config lazily,
# so loading here covers the whole run). Real environment variables still take precedence.
load_dotenv()

from agent import build_agent

DEMO_TASKS = [
    # Single country profile that interacts with several tools.
    "Build a short profile of Vietnam: capital, population, main language and its most recent GDP per capita. Include the year for each statistic.",

    # Cross-country correlation -- the core skill.
    "Compare Kenya and Ethiopia on total population and latest life expectancy. "
    "Say which has the higher life expectancy and by how many years.",

    # Mix of structured + encyclopedic context.
    "For Uzbekistan, list its bordering countries (if any), its latest GDP growth rate, and a one-sentence summary of its history and culture from Wikipedia.",

    # One-line structured profile.
    "Give a one-line profile of Egypt, including its capital, population, and main language.",
]


def run_task(agent, task: str) -> None:
    ''' Run a single task with the agent and print the result. '''
    print("\n" + "=" * 80)
    print(f"Task: {task}")
    print("=" * 80)
    try:
        # 1.x agents take a messages list and return a messages list under "messages" in the result. The final answer is in the last message's content.
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        messages = result["messages"]

        # Show the full trace (human -> tool calls -> tool results -> answer) if verbose is enabled.
        # pretty_print() formats each message type, including tool calls and tool results, for easier reading.
        for  msg in messages:
            msg.pretty_print()

        print("\nANSWER:\n" + messages[-1].content)
    except Exception as exc: # keep the demo loop running on any single failure
        print(f"\nRUN FAILED - Error running task: {exc}")


def main() -> None:
    ''' Run the country-intel agent with either the built-in demo tasks or a user-supplied question. '''
    verbose = os.getenv("VERBOSE", "1") != "0"  # default is verbose, set VERBOSE=0 to hide the step-by-step execution trace
    agent = build_agent(verbose=verbose)

    if len(sys.argv) > 1:
        # User supplied a question on the command line.
        task = " ".join(sys.argv[1:])
        run_task(agent, task)
    else:
        # Run the built-in demo tasks.
        for task in DEMO_TASKS:
            run_task(agent, task)

if __name__ == "__main__":
    main()