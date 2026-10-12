-- Idempotent completed evaluations per event + judge prompt version.

CREATE UNIQUE INDEX IF NOT EXISTS retrieval_evaluations_event_prompt_completed_uidx
    ON retrieval_obs.retrieval_evaluations (event_id, event_created_at, prompt_version)
    WHERE status = 'completed';

COMMENT ON INDEX retrieval_obs.retrieval_evaluations_event_prompt_completed_uidx IS
  'Prevents duplicate successful judge runs for the same event and prompt_version.';
