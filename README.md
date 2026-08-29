# Country-Intel Agent — Training Scaffold

A minimal, safe LangChain agent for an **agent-building training exercise**. It
correlates facts about countries from four **public, keyless, no-PII** APIs, in
the spirit of a world factbook. Model is a **local Mixtral served by vLLM**.

## Why this is safe for training
Every data source is public, needs no API key, and returns **no sensitive or
personal data**:

| Tool | API | What it gives |
|------|-----|----------------|
| `get_country_profile` | REST Countries | capital, population, area, languages, currencies, ISO codes, borders |
| `get_worldbank_indicator` | World Bank | GDP, GDP/capita, population, life expectancy, growth, inflation, unemployment |
| `get_wikipedia_summary` | Wikipedia REST | short encyclopedic text for qualitative context |
| `get_number_fact` | Numbers API | trivia about a number (adds colour) |
| `get_trivia_question` | Open Trivia DB | a multiple-choice quiz question (default: geography) |

## Files
```
tools.py    the four API tools (docstrings = what the model sees)
llm.py      connects LangChain to your local Mixtral on vLLM
agent.py    builds the agent (two modes; see below)
main.py     CLI + three demo "correlate data" tasks
```

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Confirm the APIs work (no model needed)
Run the tools directly first — this isolates "is my network/APIs OK?" from "is
my model OK?":
```bash
python tools.py
```
You should see a Vietnam profile, a World Bank GDP/capita figure, a Wikipedia
extract, and a number fact.

## 3. Serve Mixtral with vLLM
Basic serving (works with the **structured** agent mode, the default):
```bash
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --port 8000
```

To use **native tool calling** (the `tool_calling` mode) you must start vLLM
with the mistral tool parser **and** a mistral tool-call chat template — the
model's default template does **not** work for tool calls under vLLM:
```bash
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --chat-template examples/tool_chat_template_mistral_parallel.jinja
```
Note: Mistral-family models are known to be unreliable at *parallel* tool calls,
so the agent instructs the model to take one action at a time.

## 4. Run the agent
```bash
# defaults: structured mode, talks to http://localhost:8000/v1
python main.py

# ask your own question
python main.py "Compare France and Germany on GDP per capita and life expectancy."

# try native tool calling (needs the vLLM flags above)
AGENT_MODE=tool_calling python main.py
```
Config via env vars (see `.env.example`): `VLLM_BASE_URL`, `VLLM_MODEL`,
`VLLM_API_KEY`, `AGENT_MODE`.

## The two agent modes
- **`structured` (default).** Prompt-based JSON-action loop. Robust when the
  model's native tool calling is weak, supports multi-argument tools (World Bank
  needs `iso3` + `indicator`), and is transparent for teaching — set
  `verbose=True` and watch each Thought / Action / Observation.
- **`tool_calling`.** Uses Mixtral's native tool calling via vLLM. More like a
  production setup, but more fragile to serve. Good for a "now do it the real
  way" follow-up lesson.

Both use `temperature=0` for reproducible, gradeable runs.

## Suggested difficulty ramp for trainees
1. **One tool.** Make the agent answer "What is the capital of Kenya?" (only
   `get_country_profile`).
2. **Two-hop.** "What is Kenya's latest GDP per capita?" — forces
   `get_country_profile` → ISO3 → `get_worldbank_indicator`.
3. **Cross-country correlation.** "Which of Kenya, Uganda, Tanzania has the
   highest life expectancy?" — loops the same tools over several entities.
4. **Graceful failure.** Ask about a made-up country and watch it recover from
   the `error` payload instead of hallucinating.
5. **Native tool calling.** Switch to `AGENT_MODE=tool_calling` and compare
   reliability and trace shape.

## Extension ideas
- **Wikidata.** Swap or add a tool that hits the Wikidata REST/SPARQL endpoint
  for structured claims (e.g. head of government, GDP by year) to complement the
  Wikipedia prose tool.
- **More indicators.** Add rows to `WORLD_BANK_INDICATORS` in `tools.py`. (Some
  series churn — e.g. World Bank's CO2 codes have changed — so verify a new code
  returns data before relying on it.)
- **Caching.** Wrap `_get` with `functools.lru_cache` or a small on-disk cache
  so repeated grading runs don't hammer the APIs.
- **Grading harness.** Since temperature is 0, you can assert on expected
  figures/years per task to auto-score trainee agents.

## Notes / gotchas
- **Numbers API is HTTP-only** — if your environment blocks plain HTTP, that
  tool will fail while the others (HTTPS) still work.
- **Open Trivia DB rate-limits to 1 request per 5 seconds per IP.** It signals
  this with `response_code: 5` (not an HTTP error), which the tool surfaces as
  an `error` telling the agent to wait. If trainees loop it rapidly they'll hit
  this — a nice teaching moment for backoff, and a reason to add caching.
- **REST Countries** returns a *list*; the tool takes the best match `[0]`.
- **World Bank** returns `[metadata, [records]]`, newest-first, and many recent
  years are `null`; the tool skips nulls and returns the latest real value, so
  always report the `year` it gives back.