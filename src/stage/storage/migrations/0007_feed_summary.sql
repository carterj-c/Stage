ALTER TABLE jobs ADD COLUMN feed_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE quarantine ADD COLUMN feed_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE quarantine ADD COLUMN verified_description TEXT NOT NULL DEFAULT '';
UPDATE jobs SET feed_summary = description, description = '', work_auth_known = CASE WHEN work_auth_flag = 1 THEN 1 ELSE 0 END WHERE source = 'hanzili' AND description <> '';
