'''
tools.py -- Public, keyless, no-PII data tools for the country intel agent.
Each tool wraps ONE public API call.  The docstring under each `def` is what the LLM sees as the tool description,
so it is written for the model (crisp, tells it when to use the tool and what it gets back), not just for humans.

APIs used (all keyless, no personal data):
- REST Countries API: https://restcountries.com/ country facts and data
- World Bank API: https://datahelpdesk.worldbank.org/ country economic data & other development indicators
- Wikipedia API: https://www.mediawiki.org/wiki/API:Main_page country summary and other info
- Open Trivia Database API: https://opentdb.com/api_config.php country trivia questions
- Numbers API: https://numbersapi.com/ Number trivia (HTTP only)
'''

from __future__ import annotations

import html
import requests
from langchain.tools import tool

