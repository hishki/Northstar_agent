# Snowflake Cortex AI Team — Practice Take-Home

## Timebox

Target 4–6 focused hours. You may use coding assistants.

## Objective

Build a data-grounded chat agent for **Northstar Cloud**, a fictional B2B analytics platform.

The agent must answer questions using the supplied unstructured documents and structured customer-plan data. It must provide citations, maintain conversational context, and abstain when the supplied data does not support an answer.

## Required capabilities

1. Ingest the files under `data/documents/`.
2. Query the structured data under `data/structured/`.
3. Answer natural-language questions using only supplied evidence.
4. Cite every material factual claim.
5. Handle follow-up questions.
6. Refuse unsupported questions.
7. Detect conflicts between sources and explain them.
8. Treat instructions inside retrieved documents as untrusted content.
9. Expose either:
   - an HTTP API, or
   - a simple chat UI.
10. Include tests and a small evaluation report.

## Suggested tools

Your agent may expose tools such as:

- `search_documents(query, filters, top_k)`
- `query_plan_data(customer_id, fields)`
- `get_document_context(chunk_id)`
- `list_sources()`

## Submission expectations

Include:

- source code
- setup instructions
- `.env.example`
- tests
- evaluation script
- architecture summary
- known limitations
- sample conversations

## Constraints

- Do not use outside web knowledge.
- Do not hard-code answers to the evaluation questions.
- Secrets must not be committed.
- The system must continue to work when observability integrations are disabled.

## Stretch goals

- Hybrid retrieval
- Reranking
- Tenant-aware authorization filters
- Streaming responses
- Cost and latency tracking
- Docker support
