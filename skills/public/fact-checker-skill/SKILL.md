---
name: fact-checker-skill
description: |
  Methodology for verifying numerical claims in research reports against
  cited sources. Used by fact-checker-sonnet and fact-checker-gpt
  subagents. Defines the per-claim verification workflow, output schema,
  and strict classification rules.
# Phase 2 Day 3 P0-2 fix: removed `allowed-tools: [...]` because
# tool_policy.py applies the union to ALL agents that load this skill
# (incl. lead agent) — stripped write_file / task / read_file etc from
# main agent's tool catalog, making it appear "not exposed". Subagent
# fact-checker-sonnet/gpt are already restricted independently via their
# config's `tool_groups: [web, verify]`, so removing this here doesn't
# change subagent behavior. Verified: 10/10 sanity Day 2 had 0 file writes,
# Day 3 single probe confirmed missing tools in request.tools.
---

# Fact-Checker Skill — Phase 1

## Purpose

Verify whether a research report's specific numerical claims are
supported by the URLs cited inline as `[citation:TITLE](URL)`.

## Phase A: Claim Extraction

Given a section of markdown text:

1. Tokenize by sentence (split on `. ` and `。`).
2. For each sentence, check if it contains BOTH:
   - A number followed by a unit (`%`, `$`, `¥`, `K`/`M`/`B`, `MTok`,
     `x`/`×` multiplier, benchmark score like `80.9%`)
   - A `[citation:TITLE](URL)` reference (within the same sentence or
     the immediately following one).
3. Pair (claim_sentence, citation_url) — call these "claims to verify".
4. Drop sentences with no citation → mark as `unverifiable_no_source` in
   the final report.

## Phase B: Tool Invocation

For each (claim_sentence, url) pair:

```
verify_numbers(
    description="<why you are verifying this claim>",
    claim_text=<the full sentence>,
    source_url=<the cited URL>,
)
```

The tool fetches the source via Jina, runs Level B (number presence)
and Level A (direction conflict) checks, and returns a structured JSON
verdict. **Respect that verdict verbatim** — do NOT re-classify based on
your own reading.

## Phase C: Verdict aggregation

Roll up per-claim results into a section-level verdict:

| Per-claim breakdown | Section verdict |
|---|---|
| All claims `supported` | `supported` |
| ≥1 `unsupported` (any reason) | `unsupported` |
| All `supported`, 1-2 `unverifiable` (fetch failed) | `pass_with_uncertainty` |
| All `unverifiable` | `unverifiable` |
| Mix of supported + minor non-numerical issue | `partial` (reserved case) |

## Phase D: Reporting (JSON only, no prose)

```json
{
  "section_id": "<as passed in>",
  "verdict": "supported | partial | unsupported | pass_with_uncertainty | unverifiable",
  "claims_total": N,
  "claims_passed": M,
  "claims_failed": K,
  "section_features": {
    "num_count": N,
    "direction_word_count": M,
    "has_recent_date_180d": true|false
  },
  "failures": [
    {
      "claim_text": "...",
      "source_url": "...",
      "fail_reason": "number_missing_in_source | direction_conflict | quote_not_in_source | source_fetch_failed",
      "suggested_fix": "..."
    }
  ]
}
```

## CRITICAL CLASSIFICATION RULE (v3 紧 prompt, PoC #4 +40pp)

- **"unsupported"** — Output this if ANY of:
  (a) the source CONTRADICTS the claim (direction / number conflict)
  (b) the source does NOT MENTION a specific number / quote / entity
      that the claim attributes to it
  (c) the claim contains a number that DIFFERS from the source by ANY
      amount (even off-by-25% — see PoC #4 case 5)
  (d) the claim attributes a statement to the source that you cannot
      find verbatim or paraphrased

- **"partial"** — Reserve ONLY for: the claim's general meaning is
  supported but one minor NON-NUMERICAL detail differs (e.g., date off
  by 1 day, slightly different category wording).

- **"supported"** — Only if every specific number, quote, attribution
  can be matched verbatim or near-verbatim. **When in doubt → "unsupported"**.

### Examples

| Claim | Source | Verdict |
|---|---|---|
| "$3 input price" | "$3.75 input price" | unsupported (rule c) |
| "X says Y" attributed to source | source mentions X but not Y | unsupported (rule b) |
| "cost decreased 8%" | "cost increased 12-27%" | unsupported (rule a) |
| "released June 20" | "released June 21" | partial (reserved) |

## Hard rules (structural enforcement)

- **DO NOT call `write_file`** (subagent's tool_groups excludes it).
- **DO NOT call `task`** (subagent cannot nest task calls).
- **DO NOT call `ask_clarification`** (subagent's `disallowed_tools`
  blocks it — prevents bail on policy conflicts; see PoC #2 gpt-5 case).
- **DO NOT skip `verify_numbers`** — the wrapper enforces ≥80% verify
  coverage; <80% triggers `invalid_verify` exit code 33.

## Multi-entity awareness (critical for second-pass auditor)

PoC #4 case 3 showed that same-provider checkers can miss multi-entity
attribution errors. If a claim mentions multiple entities each with their
own numbers (e.g., "X scored A%, Y scored B%, Z scored C%"), call
`verify_numbers` separately for EACH entity. Don't mark the whole claim
as supported just because ONE entity matches.

## When everything fails

If `verify_numbers` returns `unverifiable` (e.g., source fetch failed)
for >2 claims in a section, return verdict `unverifiable` with a list of
fetch errors. Don't fabricate verification you couldn't actually perform.
