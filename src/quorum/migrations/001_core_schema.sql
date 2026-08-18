-- 001_core_schema
--
-- The shared memory. Six tables, each one carrying a specific piece of the
-- coordination story:
--
--   workspaces      one concurrent task, and the coordination mode it runs in
--   agent_sessions  who is alive right now
--   work_units      the contended resource, with a lease
--   unit_deps       the edges the invalidation cascade walks
--   decisions       cross-cutting conclusions, embedded for semantic conflict
--   findings        discoveries, some of which invalidate finished work
--   conflict_log    the headline artifact: proof that contention happened

CREATE TABLE IF NOT EXISTS workspaces (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          STRING NOT NULL,
    task_spec     JSONB NOT NULL,
    mode          STRING NOT NULL DEFAULT 'safe',   -- safe | naive
    status        STRING NOT NULL DEFAULT 'active',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT workspaces_mode_check CHECK (mode IN ('safe', 'naive'))
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  UUID NOT NULL REFERENCES workspaces(id),
    name          STRING NOT NULL,
    status        STRING NOT NULL DEFAULT 'running',
    heartbeat_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX agent_sessions_ws_status_idx (workspace_id, status)
);

CREATE TABLE IF NOT EXISTS work_units (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id      UUID NOT NULL REFERENCES workspaces(id),
    target            STRING NOT NULL,          -- e.g. a file path
    spec              JSONB NOT NULL,
    status            STRING NOT NULL DEFAULT 'pending',
                      -- pending | claimed | done | stale | failed
    claimed_by        UUID REFERENCES agent_sessions(id),
    claim_expires_at  TIMESTAMPTZ,
    version           INT NOT NULL DEFAULT 1,
    result_ref        STRING,                   -- S3 key or local path
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX work_units_ws_status_idx (workspace_id, status),
    -- Lets the reaper find expired leases without scanning every unit.
    INDEX work_units_expiry_idx (claim_expires_at) WHERE status = 'claimed',
    CONSTRAINT work_units_status_check
        CHECK (status IN ('pending', 'claimed', 'done', 'stale', 'failed'))
);

CREATE TABLE IF NOT EXISTS unit_deps (
    unit_id            UUID NOT NULL REFERENCES work_units(id),
    depends_on_unit_id UUID NOT NULL REFERENCES work_units(id),
    PRIMARY KEY (unit_id, depends_on_unit_id),
    -- The invalidation cascade walks these edges in reverse ("who depends on
    -- the node I just invalidated?"), which the primary key alone cannot serve.
    INDEX unit_deps_reverse_idx (depends_on_unit_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  UUID NOT NULL REFERENCES workspaces(id),
    agent_id      UUID REFERENCES agent_sessions(id),
    scope         STRING NOT NULL,       -- what this decision governs
    statement     STRING NOT NULL,
    rationale     STRING,
    embedding     VECTOR(1024),
    status        STRING NOT NULL DEFAULT 'active',  -- active | superseded
    supersedes_id UUID REFERENCES decisions(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX decisions_ws_scope_status_idx (workspace_id, scope, status),
    CONSTRAINT decisions_status_check CHECK (status IN ('active', 'superseded'))
);

CREATE TABLE IF NOT EXISTS findings (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  UUID NOT NULL REFERENCES workspaces(id),
    unit_id       UUID REFERENCES work_units(id),
    content       STRING NOT NULL,
    embedding     VECTOR(1024),
    invalidates   BOOL NOT NULL DEFAULT false,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX findings_ws_created_idx (workspace_id, created_at DESC)
);

CREATE TABLE IF NOT EXISTS conflict_log (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  UUID NOT NULL REFERENCES workspaces(id),
    kind          STRING NOT NULL,   -- claim | semantic | invalidation
    agents        UUID[],
    detail        JSONB NOT NULL,
    resolution    STRING,
    detected_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at   TIMESTAMPTZ,
    INDEX conflict_log_ws_detected_idx (workspace_id, detected_at DESC),
    CONSTRAINT conflict_log_kind_check
        CHECK (kind IN ('claim', 'semantic', 'invalidation'))
);
