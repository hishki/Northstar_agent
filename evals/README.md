# Evaluation Guidance

Do not hard-code these answers.

Recommended metrics:

- retrieval recall@k
- citation correctness
- grounded-answer correctness
- unsupported-question accuracy
- prompt-injection resistance
- median and p95 latency
- token usage per answer

A strong implementation should also test multi-source questions and customer-specific overrides.
