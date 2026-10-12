-- Lease fencing, retry scheduling, and reclaim of expired running jobs.

ALTER TABLE retrieval_obs.evaluation_jobs
    ADD COLUMN IF NOT EXISTS lease_token UUID,
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS max_attempts INTEGER;

UPDATE retrieval_obs.evaluation_jobs
SET max_attempts = 3
WHERE max_attempts IS NULL;

ALTER TABLE retrieval_obs.evaluation_jobs
    ALTER COLUMN max_attempts SET DEFAULT 3;

UPDATE retrieval_obs.evaluation_jobs
SET max_attempts = 3
WHERE max_attempts IS NULL;

ALTER TABLE retrieval_obs.evaluation_jobs
    ALTER COLUMN max_attempts SET NOT NULL;

COMMENT ON COLUMN retrieval_obs.evaluation_jobs.lease_token IS
  'Fencing token: worker must present token to complete or extend lease.';
COMMENT ON COLUMN retrieval_obs.evaluation_jobs.next_attempt_at IS
  'Worker must not claim job before this timestamp (exponential backoff).';

CREATE INDEX IF NOT EXISTS evaluation_jobs_claim_idx
    ON retrieval_obs.evaluation_jobs (evaluation_version, status, next_attempt_at, lease_expires_at, created_at)
    WHERE status IN ('pending', 'leased', 'running');
