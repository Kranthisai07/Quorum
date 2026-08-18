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

**AWS retired the "Model access" page.** Serverless foundation models are now
enabled automatically, per account, in every commercial region, the first time
you invoke them. There is no longer a list of checkboxes to tick, and older
guides telling you to "request model access" are describing a console page that
no longer exists.

Two things still gate you, and only one is likely to bite:

### Anthropic models may ask for use-case details

First-time users of Anthropic models may have to submit a short use-case form
before the model will answer. Nothing surfaces this from the API -- a call just
fails with `AccessDenied` -- so clear it from the console *before* wiring up
credentials:

1. Console → **Bedrock** → **Model catalog** → pick the Claude model in
   `QUORUM_BEDROCK_TEXT_MODEL`.
2. Open it in the **playground** and send one message.
3. If a use-case form appears, fill it in. Approval is usually immediate.

That single playground message doubles as the "first invocation" that enables
the model account-wide, so it settles both questions at once. Titan embeddings
have no such gate and enable silently.

### Accepting the agreement needs Marketplace permissions, not Bedrock ones

The Anthropic agreement is an **AWS Marketplace offer** (its ARN looks like
`arn:aws:catalog:us-east-1::offer/offer-...`). Accepting it therefore needs
Marketplace rights, which `AmazonBedrockFullAccess` does not grant. An IAM user
scoped for inference hits `AccessDeniedException` on this step even though every
Bedrock permission it needs is present.

**Accept it once as the account root user**, in the console. That is the right
principal for a commercial agreement, it is a one-time account-wide action, and
it leaves the inference user scoped to inference — which is where you want it.
Afterwards the restricted key works with no policy change at all.

If you would rather grant it to an IAM user instead, the missing actions are:

```json
{
  "Effect": "Allow",
  "Action": [
    "aws-marketplace:Subscribe",
    "aws-marketplace:ViewSubscriptions"
  ],
  "Resource": "*"
}
```

Note that `aws-marketplace:Subscribe` cannot be scoped to a single model, which
is a good reason to prefer the root-once approach and keep it off the dev user.

### Marketplace-served models need one privileged invocation

Models served through AWS Marketplace need a user with AWS Marketplace
permissions to invoke them once, which enables them account-wide for everyone
else. Neither Claude nor Titan is affected on a standard account; this matters
only in an organisation where a restricted role is the first to call a
Marketplace model.

### If Claude Opus 5 is not available on your account

Claude Opus 4.8, Sonnet 5, and Haiku 4.5 are open to all Bedrock customers, so
any of these is a one-line change:

```ini
QUORUM_BEDROCK_TEXT_MODEL=anthropic.claude-opus-4-8
# or anthropic.claude-sonnet-5
# or anthropic.claude-haiku-4-5   (cheapest; fine for the classifier)
```

Nothing else in the codebase changes -- the model id is configuration.

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

### The four access gates

Bedrock answers an unauthorised call with `not available for this account`,
which is **four different problems wearing one message**. `quorum bedrock check`
queries `GetFoundationModelAvailability` and reports which gate actually failed:

| Gate | Meaning when it fails | Fix |
|---|---|---|
| `region` | The model is not offered in this region | Change `AWS_REGION`, or pick a model that is offered there |
| `authorization` | Your IAM policy does not allow this model | Add `bedrock-mantle:CreateInference` / `bedrock:InvokeModel` on the model ARN |
| `entitlement` | The account is not entitled to the model | Some models are gated per account; pick an open one |
| `agreement` | The provider's use-case agreement is not accepted | Open the model in the Bedrock playground and submit the use-case form |

Example of a real diagnosis — everything green except the last, so the use-case
form is the only blocker and no amount of IAM editing will help:

```
gates  anthropic.claude-opus-5
       region=AVAILABLE  authorization=AUTHORIZED
       entitlement=AVAILABLE  agreement=NOT_AVAILABLE
```

Note that `ListFoundationModels` is not a substitute for this. It lists models
that *exist*, with `ACTIVE` status, regardless of whether your account may call
them — `anthropic.claude-opus-5` shows as `ACTIVE` while returning 403.

### Throttling on a new account

`ThrottlingException` on every attempt, with no successful call, is usually not
rate limiting -- it is a brand-new account whose Bedrock quotas have not been
provisioned yet. Check the four gates first: if they are all green and calls
still throttle, wait and retry rather than changing anything. Quotas normally
appear within a few hours of account activation.

Read the failures literally — each names its own endpoint:

| Message | Meaning |
|---|---|
| `Could not resolve AWS credentials from session` | No credentials reached the Claude client. Check `.env`. |
| `Unable to locate credentials` | Same, from boto3. |
| `AccessDeniedException` on the text path | Credentials fine. Either the IAM policy is missing `bedrock-mantle:CreateInference`, or Anthropic's first-use gate has not been cleared — open the model in the Bedrock playground once (see §2). |
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
