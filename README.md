# Northstar Cloud Support Agent

A grounded RAG chat agent for **Northstar Cloud**, a fictional B2B analytics platform. It answers internal support-agent questions using *only* the supplied product/policy/security/incident-response documents (`data/documents/`) and customer/plan records (`data/structured/`), citing every material claim, resolving follow-ups, preferring newer policy versions on conflict, respecting customer-specific contract overrides, abstaining when unsupported, and treating a deliberately-planted prompt injection (`data/documents/migration_guide.md`) as inert data.

> The original take-home brief is preserved verbatim at [`TAKE_HOME_BRIEF.md`](TAKE_HOME_BRIEF.md) / [`ASSIGNMENT.md`](ASSIGNMENT.md). This file documents what was actually built.

## Stack

| Concern | Technology | Config key |
|---|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) (agent ↔ tools loop + citation-validation node) | — |
| LLM | Local, via [Ollama](https://ollama.com) — default `qwen2.5:7b-instruct` | `llm.provider`, `llm.model` |
| Vector store | [Qdrant](https://qdrant.tech) (own Docker service, or `:memory:` for tests/dev) | `vector_store.provider` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2` by default) | `embeddings.model` |
| Lexical retrieval | `rank_bm25` + stemming, fused with embeddings via Reciprocal Rank Fusion, reranked by a cross-encoder | `retrieval.mode`, `retrieval.reranker.*` |
| Auth | Bearer API key per support agent (`app/security/auth.py`) | `auth.*`, `AGENT_API_KEYS` env var |
| Tracing | [Langfuse](https://langfuse.com) (per-turn trace + per-tool-call span; no-ops gracefully if unconfigured) | `observability.*`, `LANGFUSE_*` env vars |
| API | FastAPI (`POST /chat`, `POST /chat/stream` for SSE progress events, `GET /sources`) | — |
| Packaging | Docker Compose (`qdrant` + `api` by default; optional fully-dockerized `ollama`) | — |

**Every one of these is a config field, not a hardcoded import** — see [Configuration](#configuration).

> **GPU note.** Docker Desktop cannot pass a host GPU through to containers (true on macOS for Apple Silicon, and relevant on any host without `nvidia-container-toolkit`). Running Ollama *inside* Docker means CPU-only inference — for a 7B model that's roughly 1-3 minutes per question. The default setup below runs Ollama natively instead (GPU-accelerated via Metal on Apple Silicon), which is 10-20x+ faster, and only `qdrant` + `api` stay in Docker. A fully-dockerized, zero-native-installs option is still available — see [Alternative: fully-dockerized Ollama](#alternative-fully-dockerized-ollama-slower).

## Quickstart (recommended): native Ollama + Docker for qdrant/api

```sh
# 1. Install and start Ollama natively, then pull the default model (~5GB)
brew install ollama              # or download from https://ollama.com
brew services start ollama       # runs in the background, GPU-accelerated on Apple Silicon
ollama pull qwen2.5:7b-instruct

# 2. Bring up qdrant + api (talks to the host's Ollama via host.docker.internal)
docker compose up --build
```

Once it's up, set an agent API key (see `.env.example` for the `AGENT_API_KEYS` format — `key:agent_id` pairs) and pass it as a bearer token:

```sh
curl -s localhost:8000/chat -H 'content-type: application/json' -H 'authorization: Bearer sk-dev-key' -d '{
  "message": "Does Cedar Finance have a dedicated TAM?",
  "conversation_id": "demo-123",
  "customer_id": "CUST-1003"
}' | python3 -m json.tool

curl -s localhost:8000/sources -H 'authorization: Bearer sk-dev-key' | python3 -m json.tool

# SSE progress-event stream (see DESIGN.md -- this streams which tool is
# running while the turn is in flight, not the answer text token-by-token):
curl -N -s localhost:8000/chat/stream -H 'content-type: application/json' -H 'authorization: Bearer sk-dev-key' -d '{
  "message": "Does Cedar Finance have a dedicated TAM?",
  "conversation_id": "demo-124",
  "customer_id": "CUST-1003"
}'
```

`/chat`'s response shape matches [`sample_api_contract.json`](sample_api_contract.json). Set `auth.enabled: false` in `config/default.yaml` (or leave `AGENT_API_KEYS` unset with auth left enabled, which 401s) to skip auth locally — see `DESIGN.md`'s Security considerations for the authorization model.

## Quickstart — fully local (no Docker at all)

Requires Python 3.9+ and native Ollama (as above).

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Qdrant: either run it too (docker run -p 6333:6333 qdrant/qdrant) and set
# QDRANT_URL=http://localhost:6333, or leave QDRANT_URL unset -- the app
# falls back to an in-memory Qdrant instance, rebuilt at startup.
cp .env.example .env   # adjust if needed

.venv/bin/uvicorn app.api:app --reload
```

## Alternative: fully-dockerized Ollama (slower)

For a true zero-native-installs demo (or a host without a GPU worth using anyway), bring up the old fully-containerized Ollama with the `dockerized-ollama` Compose profile instead:

```sh
OLLAMA_BASE_URL=http://ollama:11434 docker compose --profile dockerized-ollama up --build
```

This starts `qdrant`, `ollama` (a one-shot `ollama-pull` service pulls `qwen2.5:7b-instruct` into it on first boot), and `api`, all in Docker — but CPU-only inference inside the container, ~1-3 minutes per question rather than seconds.

## Tests

Every module was built with its own numbered test file so a stage can be run and debugged in isolation:

```sh
# One module at a time, e.g.:
.venv/bin/pytest tests/test_06_search_hybrid.py -v

# Everything that doesn't need a live Ollama/Qdrant service (this is what CI should run):
.venv/bin/pytest tests/ -m "not live"

# Live end-to-end smoke tests (needs a reachable Ollama, ideally Qdrant too):
docker compose up -d qdrant ollama    # or: ollama serve && ollama pull qwen2.5:7b-instruct
.venv/bin/pytest tests/ -m live
```

| File | Covers |
|---|---|
| `test_00_config.py` / `test_00_interfaces.py` | Config loading, LLM-provider dispatch, fakes satisfy the Protocol contracts |
| `test_01_structured_data.py` .. `test_03_structured_tool.py` | CSV loading, markdown chunking, `query_plan_data`/`list_sources` |
| `test_04_embeddings.py` .. `test_07_conflict.py` | BM25, embeddings/Qdrant, hybrid RRF fusion + recall@k against real doc content, recency/conflict tagging |
| `test_08_injection_wrapping.py` | Untrusted-content wrapping, injection-phrase heuristic |
| `test_09_system_prompt.py` | Grounding-contract policy clauses present |
| `test_10_orchestrator_unit.py` / `test_10_orchestrator_live.py` | LangGraph loop, citation verification, abstention, loop-limit guard (scripted, no model) / real end-to-end (live, skipped by default) |
| `test_11_api.py` | `POST /chat` / `GET /sources` shape (orchestrator stubbed via FastAPI dependency override) |
| `test_12_eval_harness.py` | Every eval metric's math, against synthetic fixtures |
| `test_13_reranker.py` | Cross-encoder reranker promotes the correct chunk over the fused BM25+embeddings pool; disabled/failed-load falls back to plain RRF |
| `test_14_auth.py` | `require_agent` 401 cases, valid-key success, `auth.enabled=False` bypass, conversation-id agent-namespacing isolation, Langfuse tracing invoked with correct fields, graceful no-op with Langfuse unconfigured |
| `test_15_streaming.py` | `POST /chat/stream` SSE event shape, parity with non-streaming `/chat`'s final response, auth enforcement on the streaming endpoint |

## Evaluation

```sh
# Needs a live Ollama (+ ideally Qdrant) reachable, per the Quickstart above.
.venv/bin/python evals/run_eval.py            # heuristic grounded-answer grading
.venv/bin/python evals/run_eval.py --judge     # also use the LLM itself as a judge
```

Writes `evals/results.json` (raw per-question data) and `evals/report.md` (the human-readable report: retrieval recall@k, citation correctness, grounded-answer correctness, abstention accuracy, prompt-injection resistance, latency, token usage). See [`evals/README.md`](evals/README.md) for the metric definitions.

## Configuration

Every technology choice lives in [`config/default.yaml`](config/default.yaml); secrets/connection endpoints come from environment variables (see [`.env.example`](.env.example)). To migrate a piece of the stack later:

1. Implement the relevant `Protocol` from `app/interfaces.py` (`StructuredStore`, `DocumentStore`, `Retriever`, or `Sanitizer`) with the new technology.
2. Add a branch for it in that package's `create_*` factory (e.g. `app/retrieval/__init__.py::create_retriever`) or in `app/factory.py::build_llm` for a new model provider.
3. Flip the corresponding field in `config/default.yaml`.

No changes needed to the LangGraph orchestrator, the API, or existing tests — they only ever depend on the Protocols.

**Prompt text is also externalized**, separate from the technology config: the grounding system prompt lives at [`prompts/system_prompt.md`](prompts/system_prompt.md) and the 5 tool descriptions at [`prompts/tool_descriptions.yaml`](prompts/tool_descriptions.yaml) — edit either without touching any Python (`app/agent/prompts.py` loads both; `app/agent/system_prompt.py` and `app/agent/tools.py` just consume them).

## Project layout

```
config/default.yaml     technology choices (llm/vector_store/embeddings/retrieval/chunking/sanitizer)
prompts/
  system_prompt.md       the grounding contract given to the model
  tool_descriptions.yaml the 5 tools' model-facing descriptions
app/
  schemas.py             shared pydantic models
  interfaces.py          Protocol contracts (StructuredStore, DocumentStore, Retriever, Sanitizer)
  config.py, factory.py  config loading + concrete-implementation construction
  fakes.py               in-memory Protocol implementations used by tests
  data/                  CSV loader + markdown chunker
  retrieval/             BM25 + Qdrant embeddings + RRF fusion + recency/conflict tagging
  security/               untrusted-content wrapping + injection heuristic
  agent/                 prompts.py (loader), tools.py (incl. submit_answer),
                          LangGraph orchestrator (AgentRuntime)
  api.py                  FastAPI app
evals/
  questions.jsonl         the 15 eval questions
  run_eval.py             harness -> results.json + report.md
tests/                    test_00_*.py .. test_12_*.py, one stage at a time
DESIGN.md                 architecture / grounding / citation / security / scaling / limitations
sample_conversations.md   example conversations (see file header for how they were produced)
```

## Known limitations

See [DESIGN.md](DESIGN.md#limitations) for the full list. The short version: authorization is broad (any authenticated agent can query any customer, audit-logged but not row-restricted — matches the actual domain, see DESIGN.md for why); citation-source attribution for structured data collapses to `customers.csv` even though `plans.csv` also contributes fields; the injection heuristic is phrase/regex-based (won't catch obfuscated/translated attacks, though the untrusted-content wrapping and the model's own instruction-following are the real backstop); grounded-answer correctness defaults to a keyword-overlap heuristic against the eval questions' `notes` field rather than an exact-answer check, since the assignment forbids hard-coded expected answers; token-usage reporting depends on the LLM provider populating `usage_metadata`, which isn't guaranteed on every Ollama/model version; streaming is progress-events only, not token-level answer streaming; auth is a flat shared-key scheme, not a real identity provider.
