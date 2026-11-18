-- Preserve unverified text for recovery without presenting a summary as a full posting.
CREATE TABLE oracle_unverified_descriptions (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
CREATE TABLE oracle_full_details (
    id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE
);
-- Prior Oracle detail outcomes were emitted only by the validated ById adapter.
INSERT INTO oracle_full_details (id)
SELECT j.id FROM jobs j JOIN detail_fetches d ON d.id = j.id
WHERE j.source = 'oracle_cloud' AND j.description <> ''
  AND d.source = 'oracle_cloud' AND d.resolved = 1 AND d.failed = 0;
INSERT INTO oracle_unverified_descriptions (id, description)
SELECT j.id, j.description FROM jobs j
WHERE j.source = 'oracle_cloud' AND j.description <> ''
  AND NOT EXISTS (SELECT 1 FROM oracle_full_details f WHERE f.id = j.id);
UPDATE jobs SET description = '', work_auth_known = 0, degree_requirement = 'unknown'
WHERE id IN (SELECT id FROM oracle_unverified_descriptions);
