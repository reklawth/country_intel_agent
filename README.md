# Country-Intel Agent — Training Scaffold

A minimal, safe LangChain agent for an **agent-building training exercise**. It
correlates facts about countries from public APIs, in the spirit of a world
factbook. Model is a **local Mistral-7B-Instruct-v0.3 served by vLLM**.

## Tools
Every data source is public and returns **no sensitive or personal data**. All are
keyless except REST Countries (see note):

| Tool | API | What it gives |
|------|-----|----------------|
| `get_country_profile` | REST Countries (v5) | capital, population, area, languages, currencies, ISO codes, borders |
| `get_world_bank_indicator` | World Bank | GDP, GDP/capita, population, life expectancy, growth, inflation, unemployment, exports |
| `get_wikipedia_summary` | Wikipedia REST | short encyclopedic text for qualitative context |
| `get_trivia_question` | Open Trivia DB | a multiple-choice quiz question by category + difficulty |

> **REST Countries v5 requires an API key.** The old keyless v3.1 API was deprecated,
> so `get_country_profile` reads a free key from the `REST_COUNTRIES_KEY` environment
> variable (sign up at <https://restcountries.com/sign-up>). The other three tools stay
> keyless.
>
> Open Trivia DB has **no country-specific categories**, so it can only supply general
> questions by category — it can't answer "a trivia question about <country>".

## Files
```
tools.py    the four API tools (docstrings = what the model sees)
llm.py      connects LangChain to your local Mistral-7B on vLLM
agent.py    builds the agent (create_agent + ReliableToolCallMiddleware)
main.py     CLI + four demo "correlate data" tasks
```

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Confirm the APIs work (no model needed)
Call the tools directly first — this isolates "is my network/APIs OK?" from "is
my model OK?" (set `REST_COUNTRIES_KEY` first; see [Tools](#tools)):
```bash
python -c "from tools import get_country_profile, get_world_bank_indicator as wb; \
print(get_country_profile.invoke({'name': 'Vietnam'})); \
print(wb.invoke({'iso3': 'VNM', 'indicator_name': 'GDP per capita (current US\$)'}))"
```
You should see a Vietnam profile and its latest World Bank GDP/capita figure.

## 3. Serve the model with vLLM
The agent uses the model's **native tool calling**, so vLLM must be started with the
mistral tool parser and auto tool choice:
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```
No `--chat-template` flag is needed — the v0.3 tokenizer ships a tool-compatible
template.

> **Use v0.3, not Mixtral-8x7B-Instruct-v0.1.** The mistral tool parser needs the
> tool-call tokens (`[TOOL_CALLS]`, `[AVAILABLE_TOOLS]`) that only exist in the **v3
> tokenizer** (32768-token vocab). Mixtral-8x7B-Instruct-v0.1 and its AWQ quants use
> the older v1 tokenizer (32000-token vocab) with **no tool tokens**, so vLLM fails
> with *"Mistral Tool Parser could not locate the tool call token in the tokenizer."*

Mistral-7B-v0.3 runs comfortably on a single 24 GB GPU at full precision. To serve it
in Docker (the model repo is gated — accept the license and pass an `HF_TOKEN`):
```bash
docker run --rm --runtime nvidia --gpus '"device=0"' --ipc=host -p 8000:8000 \
  -e HF_TOKEN=hf_your_token \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.3 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser mistral
```

## Model and its constraints
**Model:** `mistralai/Mistral-7B-Instruct-v0.3`, served locally by vLLM at
`temperature=0` for reproducible runs.

**Why this model**
- It carries the **v3 tokenizer** with the tool-call tokens vLLM's mistral parser
  requires; Mixtral-8x7B-Instruct-v0.1 does not and cannot do native tool calling here.
- At 7B it fits a single 24 GB GPU in full precision.

**Known behavioral limits** (it is a small local model):
- **Single-tool / single-country tasks are reliable** — e.g. a country profile, or one
  World Bank indicator after resolving the ISO3 code.
- **Multi-step / multi-part tasks are best-effort.** The model tends to *satisfice*:
  after the first tool call it may answer the rest from memory instead of calling the
  next tool, and it sometimes writes a tool call as JSON text rather than emitting it
  natively. `ReliableToolCallMiddleware` in `agent.py` mitigates this — it forces the
  first tool call and promotes text-written tool calls into real ones — but does not
  fully solve it.
- **Prompt-sensitive.** Even at `temperature=0`, small wording changes can flip whether
  it calls a tool on a given turn.
- **Heavy tasks are slow.** Multi-country comparisons make several calls plus a long
  synthesis; the per-call timeout is `VLLM_TIMEOUT` (default 180s).

For reliable multi-tool orchestration, serve a stronger tool-calling model (the code is
model-agnostic — just point `VLLM_MODEL` at it).

## 4. Run the agent
```bash
# defaults: talks to http://localhost:8000/v1
python main.py

# ask your own question
python main.py "Compare France and Germany on GDP per capita and life expectancy."
```
Config via env vars (see `.env`): `VLLM_HOST`, `VLLM_MODEL`, `VLLM_API_KEY`,
`VLLM_TIMEOUT`, and `REST_COUNTRIES_KEY` (required — REST Countries v5 needs a key).
`VERBOSE=0` hides the step-by-step execution trace.

## How the agent works
Built with LangChain 1.x `create_agent` (a LangGraph agent) using the model's
**native tool calling** via vLLM. A custom `ReliableToolCallMiddleware` (in `agent.py`)
compensates for the small model: it forces the first turn to be a real tool call and
promotes any tool call the model writes as JSON text into an actual call. See
[Model and its constraints](#model-and-its-constraints) for what this does and does not
fix. Runs at `temperature=0` for reproducible, gradeable output; pass `verbose=True`
(or leave `VERBOSE=1`) to watch each step of the graph.

## Suggested difficulty ramp for trainees
1. **One tool.** Make the agent answer "What is the capital of Kenya?" (only
   `get_country_profile`).
2. **Two-hop.** "What is Kenya's latest GDP per capita?" — forces
   `get_country_profile` → ISO3 → `get_world_bank_indicator`.
3. **Cross-country correlation.** "Which of Kenya, Uganda, Tanzania has the
   highest life expectancy?" — loops the same tools over several entities. (Note:
   multi-country tasks are best-effort on the 7B model; see constraints above.)
4. **Graceful failure.** Ask about a made-up country and watch it recover from
   the `error` payload instead of hallucinating.

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
- **REST Countries v5 needs a key** and its response is JSON:API-shaped
  (`data.objects[...]`); the tool takes the best match and flattens it. A missing
  `REST_COUNTRIES_KEY` makes `get_country_profile` return an `error` (the demo key
  `rc_live_demo` only ever returns Canada, so it is not used).
- **Open Trivia DB rate-limits to 1 request per 5 seconds per IP.** It signals
  this with `response_code: 5` (not an HTTP error), which the tool surfaces as
  an `error` telling the agent to wait. If trainees loop it rapidly they'll hit
  this — a nice teaching moment for backoff, and a reason to add caching.
- **World Bank** returns `[metadata, [records]]`, newest-first, and many recent
  years are `null`; the tool skips nulls and returns the latest real value, so
  always report the `year` it gives back.