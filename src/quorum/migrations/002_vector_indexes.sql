-- 002_vector_indexes
--
-- quorum:no-transaction
--
-- The distributed vector index is the primitive behind semantic conflict
-- detection. Keyword matching cannot tell that "standardise on httpx" and
-- "keep using requests for the transport layer" are the same argument, so the
-- pre-commit check on every decision is a nearest-neighbour query, not a LIKE.
--
-- Both indexes lead with workspace_id. CockroachDB supports prefix columns on a
-- vector index, so the ANN search is scoped to a single workspace instead of
-- searching every workspace and filtering afterwards.
--
-- Opclass is vector_cosine_ops (<=>) because Titan Text Embeddings V2 returns
-- normalised vectors and we care about direction, not magnitude. Cosine
-- distance in [0, 2]; cosine similarity is 1 - distance.
--
-- Verified against CockroachDB v26.2.5: `feature.vector_index.enabled` defaults
-- to true, so no cluster setting is required on a modern cluster. Older
-- clusters (v25.2 - v25.4) need:
--     SET CLUSTER SETTING feature.vector_index.enabled = true;
-- which `quorum db init` issues defensively.

CREATE VECTOR INDEX IF NOT EXISTS decisions_embedding_idx
    ON decisions (workspace_id, embedding vector_cosine_ops);

CREATE VECTOR INDEX IF NOT EXISTS findings_embedding_idx
    ON findings (workspace_id, embedding vector_cosine_ops);
