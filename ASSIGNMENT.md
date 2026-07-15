# Assignment

Northstar Cloud's support team needs an internal assistant. Support agents ask about product policies, security commitments, incident procedures, and customer-specific entitlements.

Build a grounded chat agent that can answer questions from:

- product and policy documents
- security documentation
- incident-response documentation
- pricing and service-plan data
- customer subscription records

## Functional requirements

### Grounded answers

Every answer must be supported by the supplied data. Include citations with:

- source filename
- page or section when available
- short supporting excerpt

### Structured-data questions

The agent must answer customer-specific questions such as:

- What plan is customer `CUST-1003` on?
- How many support hours are included?
- Does that customer have a dedicated technical account manager?
- What is the customer's data-retention period?

### Follow-up questions

The agent must resolve questions such as:

> What is the refund window?

followed by:

> Does that apply to enterprise customers?

### Abstention

For unsupported questions, respond clearly that the available sources do not provide the answer.

### Conflict handling

When two sources conflict, prefer a newer effective date when clearly stated. Otherwise present the conflict and cite both sources.

### Prompt-injection resistance

One supplied document contains malicious instructions. The assistant must ignore those instructions and treat them only as document content.

## Deliverables

1. Working application
2. README
3. Tests
4. Evaluation report
5. Short design note covering:
   - architecture
   - retrieval approach
   - grounding strategy
   - citation strategy
   - security considerations
   - scaling approach
   - limitations
