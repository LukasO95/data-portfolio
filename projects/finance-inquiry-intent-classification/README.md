# Finance Inquiry Intent Classification

## Overview
This project implements a production-oriented NLP pipeline to classify free-text
customer inquiries in the **financial services domain**.

Incoming messages from a web form are automatically classified into:
- **product** (sales / advisory requests)
- **service** (existing contracts, support, documents, access issues)
- **unclear** (insufficient information → manual clearing)

The resulting classification is forwarded to a CRM system to enable automated routing
to **Sales**, **Customer Support**, or a **Clearing Queue**.

---

## Business Problem
Financial institutions receive large volumes of unstructured customer inquiries.
Manual triage is:
- slow
- error-prone
- expensive

Misrouting (e.g. sales requests sent to support) leads to inefficiency
and poor customer experience.

---

## Solution
- Prompt-based intent classification using an LLM
- Strict **JSON output contract** for semantic classification
- Defensive normalization of LLM output
- Deterministic routing logic implemented in code
- Confidence-based fallback to manual clearing
- Privacy-aware preprocessing (PII masking)
- Client-side rate limiting for real API usage

---

## Architecture

Web Form  
→ PII Masking  
→ LLM Classification (Mock or OpenAI)  
→ Output Normalization  
→ JSON Schema Validation  
→ Routing Policy  
→ CRM Payload

---

## Classification Labels

| Label | Description |
|------|-------------|
| product | Interest in financial products (mortgages, loans, savings, funding) |
| service | Existing contracts, documents, payments, login, technical issues |
| unclear | No clear intent or insufficient information |

---

## Output (CRM Payload)

The pipeline produces a strict, CRM-ready JSON object:

| Field | Description |
|-----|-------------|
| `label` | `product`, `service`, `unclear` |
| `confidence` | 0.0–1.0 heuristic confidence score |
| `handoff` | `sales`, `support`, `clearing` (set by routing policy) |
| `product_area` | `mortgage`, `savings`, `home_savings`, `other`, `none` |
| `service_area` | `existing_contract`, `online_banking`, `documents`, `payments`, `complaints`, `other`, `none` |
| `needs_human` | `true` if routed to manual clearing |
| `reason_codes` | short diagnostic reason codes |
| `classifier_version` | business logic version |

---

## Output Validation & Safety
LLM outputs are first **normalized defensively** to correct minor inconsistencies
(e.g. invalid category values) and are then validated against a strict JSON Schema.

If validation fails or a runtime error occurs, the inquiry is **safely routed
to a manual clearing queue**, preventing misclassification or system errors.

---

## Quickstart (Mock Mode — no API key required)

Run the full pipeline without external APIs:

```bash
python -m src.main --provider mock "Brauche Kredit für Modernisierung unseres Hauses"
```
The mock mode is a demonstration-only fallback and does not reflect real classification quality.

----

## Run with OpenAI (API Key required)

Requirements:
- OpenAI API key
- active API billing (Pay-as-you-go)

Create `.env` in project root:
```bash
OPENAI_API_KEY=sk-...
```

Run:
```bash
python -m src.main --provider openai "Brauche Kredit für Modernisierung unseres Hauses"
```
ChatGPT subscriptions are not valid for API usage.

----

## Evaluation
A lightweight evaluation script runs the pipeline on sample inquiries:
```bash
LLM_PROVIDER=openai python -m src.eval
```
Reported metrics:
- overall accuracy
- product vs. service accuracy
- unclear routing rate

----

## Tests
Unit tests focus on deterministic components and mock-based integration.
```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
Covered:
- PII masking
- normalization rules
- routing policy
- end-to-end pipeline (mock mode)

----

## Cost & Rate Control
- provider-side quota limits (OpenAI dashboard)
- client-side rate limiting (real API only)
- early exit for empty text to manual clearing

The system prioritizes **safe routing over forced classification**.

----

## Versioning
`classifier_version` identifies the business logic version (prompt + schema + routing rules). Behavioral changes increment the version and are recorded in `CHANGELOG.md`.

----

## Tech Stack
- Python
- OpenAI LLM API
- JSON Schema (structured outputs)
- Regex-based PII masking
- pytest

----

## Disclaimer
This repository is a demonstration project.
No real customer data is used.
All examples are anonymized or synthetic.