ALTER TABLE quarantine ADD COLUMN rejection_evidence TEXT NOT NULL DEFAULT '';
UPDATE quarantine SET rejection_evidence = CASE WHEN verified_description <> '' THEN verified_description ELSE feed_summary END WHERE source = 'hanzili' AND matched_phrase NOT GLOB 'copy of *' AND (verified_description <> '' OR (reason = 'out-of-scope-degree' AND feed_summary <> ''));
UPDATE quarantine SET rejection_evidence = verified_description WHERE source = 'themuse' AND verified_description <> '' AND matched_phrase NOT GLOB 'copy of *';
