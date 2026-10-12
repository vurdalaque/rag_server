-- Pre-flight duplicate check before enforcing unique completed evaluations.
-- Safe to run when 010 already created the index (no-op if no duplicates).

DO $$
DECLARE
    conflict_count BIGINT;
BEGIN
    SELECT COUNT(*)::bigint INTO conflict_count
    FROM (
        SELECT event_id, event_created_at, evaluation_version
        FROM retrieval_obs.retrieval_evaluations
        WHERE status = 'completed'
        GROUP BY event_id, event_created_at, evaluation_version
        HAVING COUNT(*) > 1
    ) duplicates;

    IF conflict_count > 0 THEN
        RAISE EXCEPTION
            'retrieval_evaluations has % duplicate completed (event_id, event_created_at, evaluation_version) groups; resolve manually before migration 011',
            conflict_count
            USING ERRCODE = '23505';
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS retrieval_evaluations_event_eval_version_completed_uidx
    ON retrieval_obs.retrieval_evaluations (event_id, event_created_at, evaluation_version)
    WHERE status = 'completed';

COMMENT ON COLUMN retrieval_obs.retrieval_evaluations.evaluation_version IS
  'Evaluator identity: prompt_version:judge_model[+methodology_fingerprint]; matches evaluation_jobs.evaluation_version.';
