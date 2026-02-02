# Changelog

All notable behavioral changes to the finance inquiry classifier are documented here.
Versioning refers to the **business logic of the classifier** (prompt, schema, routing),
not to the underlying LLM model.

---

## v1 (2026-01-22)

### Added
- End-to-end intent classification pipeline for financial service inquiries
- Support for three intent classes: `product`, `service`, `unclear`
- Deterministic Mock LLM provider for testing and demos (no external API required)
- Client-side rate limiting for OpenAI provider
- PII masking (email, phone, IBAN, URL, ID)
- Strict JSON Schema validation to prevent invalid LLM output reaching CRM (output contract)
- Normalization layer to prevent schema violations from minor LLM output errors
- Whitelisted `reason_codes` with maximum length enforcement 
- Routing policy with CRM business rules based on LLM output 
- Samples to evaluate performance of OpenAI LLM provider
- Unit test suite (pytest) covering masking, normalization, routing, and pipeline

---

## Versioning Policy

- `classifier_version` identifies the **business logic version**
  (prompt + schema + routing rules + normalization).
- Any change that can alter classification or routing behavior increments the version.
- Technical refactoring without behavioral impact does **not** require a version bump.