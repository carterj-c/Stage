ALTER TABLE jobs ADD COLUMN feed_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE quarantine ADD COLUMN feed_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE quarantine ADD COLUMN verified_description TEXT NOT NULL DEFAULT '';
UPDATE jobs SET feed_summary = description, description = '' WHERE source = 'hanzili' AND description <> '';
