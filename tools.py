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

# One shared session plus a hard timeout so a slow endpoint will not hang the agent.

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "country-intel-agent/1.0"})
_TIMEOUT = 15  # seconds

def _get(url: str, params: dict | None = None) -> dict:
    """Helper function to make a GET request and return JSON data."""
    try:
        response = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e


# 1. REST Countries API

@tool
def get_country_profile(name: str) -> dict:
    """
    Look up core facts about a country by name (e.g. "United States", "Kenya" "Japan") from the REST Countries API.

    Returns: official name, capital, region, subregion, population, area (km2), currencies, languages, ISO codes, and bordering countries.
    Will return an 'error' key if the country is not found.

    This is to be used first to resolve a country and get its iso3 code, which can then be used to query other APIs for more data.
    Input: country name (string)
    Output: dictionary with country data (capital, population, area, region, etc.)
    """
    try:
        data = _get(f"https://restcountries.com/v3.1/name/{name}",
                    params={"fullText": "true", "fields": "name,capital,region,subregion,population,area,currencies,languages,cca2,cca3,borders"})
    except requests.HTTPError:
        return {"error": f"No country found for name: {name}"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except RuntimeError as e:
        return {"error": str(e)}
    
    if isinstance(data, list) and data:
        c = data[0] # best match
        return {
            "common_name": c["name"].get("common", ""),
            "official_name": c["name"].get("official", ""),
            "capital": (c.get("capital", [""])[0] or [None])[0],
            "region": c.get("region", ""),
            "subregion": c.get("subregion", ""),
            "population": c.get("population", 0),
            "area_km2": c.get("area", 0.0),
            "currencies": list(c.get("currencies", {}).keys()),
            "languages": list(c.get("languages", {}).values()),
            "iso2": c.get("cca2", ""),
            "iso3": c.get("cca3", ""),
            "borders": c.get("borders", []),  #ISO3 codes of bordering countries

        }
    
    return {"error": f"No country found for name: {name}"}

    