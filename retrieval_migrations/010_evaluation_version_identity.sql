-- Align evaluation results with job queue identity: prompt_version + judge_model.

ALTER TABLE retrieval_obs.retrieval_evaluations
    ADD COLUMN IF NOT EXISTS evaluation_version TEXT;

UPDATE retrieval_obs.retrieval_evaluations
SET evaluation_version = prompt_version || ':' || judge_model
WHERE evaluation_version IS NULL;

ALTER TABLE retrieval_obs.retrieval_evaluations
    ALTER COLUMN evaluation_version SET NOT NULL;

DROP INDEX IF EXISTS retrieval_obs.retrieval_evaluations_event_prompt_completed_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS retrieval_evaluations_event_eval_version_completed_uidx
    ON retrieval_obs.retrieval_evaluations (event_id, event_created_at, evaluation_version)
    WHERE status = 'completed';

CREATE INDEX IF NOT EXISTS retrieval_evaluations_evaluation_version_idx
    ON retrieval_obs.retrieval_evaluations (evaluation_version, evaluated_at DESC);

COMMENT ON COLUMN retrieval_obs.retrieval_evaluations.evaluation_version IS
  'Stable evaluator identity (prompt_version:judge_model); matches evaluation_jobs.evaluation_version.';
