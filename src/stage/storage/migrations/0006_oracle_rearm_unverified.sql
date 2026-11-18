-- A failed last reread cannot establish whether an archived body was ever full.
-- Keep it hidden and retry from a clean bounded attempt budget.
UPDATE detail_fetches SET resolved = 0, failed = 1, attempts = 0
WHERE source = 'oracle_cloud'
  AND id IN (SELECT id FROM oracle_unverified_descriptions)
  AND id IN (SELECT id FROM jobs WHERE source = 'oracle_cloud' AND description = '')
  AND NOT EXISTS (SELECT 1 FROM oracle_full_details f WHERE f.id = detail_fetches.id);
