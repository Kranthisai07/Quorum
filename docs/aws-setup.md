# AWS setup

What Quorum needs from AWS, and the least-privilege way to grant it.

Everything here is optional for development: `QUORUM_LLM_BACKEND=stub` runs the
entire engine — including the 200-iteration stress suite — offline, with no
credentials and no spend. This document is for turning the real thing on.

---

## 1. Credentials

Copy `.env.example` to `.env` and fill in the two credential lines. `.env` is
gitignored and must never be committed; `.env.example` holds placeholders only.

```ini
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
# Only for temporary / SSO credentials:
# AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
QUORUM_LLM_BACKEND=bedrock
```

Both the Claude client and boto3 resolve credentials through the standard AWS
chain, so an `~/.aws/credentials` profile or an SSO session works equally well —
in that case leave the two key lines empty and set `AWS_PROFILE` instead.

---

## 2. Bedrock model access

**This is the step that is easy to miss.** Having valid credentials does not
mean you can call a model: access is granted per model, per region, in the
console.

1. AWS Console → **Amazon Bedrock** → **Model access** (make sure the region
   selector says the region in your `.env`).
2. Request access to:
   - **Anthropic — Claude** (the model in `QUORUM_BEDROCK_TEXT_MODEL`)
   - **Amazon — Titan Text Embeddings V2**
3. Titan is normally granted instantly. Anthropic models are usually instant
   too, but some are gated.

### If Claude Opus 5 is not available in your account

`anthropic.claude-opus-5` has per-account access criteria. Claude Opus 4.8,
Claude Sonnet 5, and Claude Haiku 4.5 are open to all Bedrock customers, so any
of these is a drop-in change to one line of `.env`:

```ini
QUORUM_BEDROCK_TEXT_MODEL=anthropic.claude-opus-4-8
# or anthropic.claude-sonnet-5
# or anthropic.claude-haiku-4-5   (cheapest; fine for the classifier)
```

Nothing else in the codebase changes — the model id is configuration.

---

## 3. IAM permissions

The two Bedrock paths need **two different IAM actions**, because they are two
different services:

| Path | Service | Action |
|---|---|---|
| Claude reasoning + classification | Messages API (`bedrock-mantle`) | `bedrock-mantle:CreateInference` |
| Titan embeddings | `bedrock-runtime` | `bedrock:InvokeModel` |

A policy granting only `bedrock:InvokeModel` will pass the embedding check and
fail the reasoning check — which is exactly why `quorum bedrock check` probes
them separately.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ClaudeReasoning",
      "Effect": "Allow",
      "Action": "bedrock-mantle:CreateInference",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-*"
    },
    {
      "Sid": "TitanEmbeddings",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
    }
  ]
}
```

No wildcards on the account, no `bedrock:*`. Phase 8 adds S3 and Lambda
statements to the same policy, scoped the same way.

---

## 4. Verify

```bash
quorum bedrock check
```

Expected:

```
backend  bedrock
  region        us-east-1
  reasoning     ok      anthropic.claude-opus-5
                via Messages API (bedrock-mantle)
  embeddings    ok      amazon.titan-embed-text-v2:0
                via boto3 bedrock-runtime InvokeModel
  dimensions    1024 vs VECTOR(1024) in schema  match
```

Read the failures literally — each names its own endpoint:

| Message | Meaning |
|---|---|
| `Could not resolve AWS credentials from session` | No credentials reached the Claude client. Check `.env`. |
| `Unable to locate credentials` | Same, from boto3. |
| `AccessDeniedException` on the text path | Credentials fine; missing `bedrock-mantle:CreateInference`, or model access not granted. |
| `AccessDeniedException` on the embed path | Credentials fine; missing `bedrock:InvokeModel` on the Titan model. |
| `ValidationException` naming the model id | Wrong id, or the model is not offered in this region. |
| **`dimensions … MISMATCH`** | Titan returned a width the schema does not have. Fix `QUORUM_EMBED_DIM`, or add a migration changing `VECTOR(n)`. **Resolve before Phase 4** — the semantic conflict index depends on it. |

---

## 5. Cost

Phase 4 onward makes real calls. Rough shape for the bundled 16-unit fixture:

- **Embeddings** — one Titan call per decision and per finding. Titan Text
  Embeddings V2 is inexpensive; a full run is fractions of a cent.
- **Reasoning** — one Claude call per work unit, sending one source file
  (docker-py's largest is ~1,350 lines). Sixteen units per run.
- **Classification** — one Claude call only when a decision has a near-neighbour
  above the similarity threshold. The ANN query narrows thousands of decisions
  to a handful first, so the model is never asked to scan the workspace.

To rehearse the full pipeline without spend, run with
`QUORUM_LLM_BACKEND=stub`. Everything works; only the model quality is fake, and
the stub's limitations are documented in
[architecture.md](architecture.md#42-semantic-guard-ann-pre-check-then-a-judge).
