-- retrieval_evaluations: LLM-judge outputs linked to events (no FK to hypertable).

CREATE TABLE IF NOT EXISTS retrieval_obs.retrieval_evaluations (
    evaluation_id       UUID        NOT NULL DEFAULT gen_random_uuid(),
    event_id            UUID        NOT NULL,
    event_created_at    TIMESTAMPTZ NOT NULL,

    evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    judge_model         TEXT        NOT NULL,
    prompt_version      TEXT        NOT NULL,

    per_chunk_scores    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    context_sufficiency JSONB       NOT NULL DEFAULT '{}'::jsonb,
    query_classification JSONB      NOT NULL DEFAULT '{}'::jsonb,
    aggregated_metrics  JSONB       NOT NULL DEFAULT '{}'::jsonb,

    explanation         TEXT,
    status              TEXT        NOT NULL DEFAULT 'completed',
    diagnostics         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    truncation_info     JSONB       NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT retrieval_evaluations_pkey PRIMARY KEY (evaluation_id),
    CONSTRAINT retrieval_evaluations_status_chk
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    CONSTRAINT retrieval_evaluations_event_time_chk
        CHECK (event_created_at IS NOT NULL)
);

COMMENT ON TABLE retrieval_obs.retrieval_evaluations IS
  'Judge evaluation for a retrieval event. Logical ref (event_id, event_created_at); no FK to hypertable.';

COMMENT ON COLUMN retrieval_obs.retrieval_evaluations.event_created_at IS
  'Must match retrieval_events.created_at for the referenced event_id (application-enforced).';

CREATE INDEX IF NOT EXISTS retrieval_evaluations_event_judge_prompt_idx
    ON retrieval_obs.retrieval_evaluations (
        event_id,
        event_created_at,
        judge_model,
        prompt_version,
        evaluated_at DESC
    );

CREATE INDEX IF NOT EXISTS retrieval_evaluations_event_lookup_idx
    ON retrieval_obs.retrieval_evaluations (event_id, event_created_at);

CREATE INDEX IF NOT EXISTS retrieval_evaluations_evaluated_at_idx
    ON retrieval_obs.retrieval_evaluations (evaluated_at DESC);

CREATE INDEX IF NOT EXISTS retrieval_evaluations_status_evaluated_idx
    ON retrieval_obs.retrieval_evaluations (status, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS retrieval_evaluations_judge_evaluated_idx
    ON retrieval_obs.retrieval_evaluations (judge_model, evaluated_at DESC);
