ALTER TABLE jobs ADD COLUMN work_auth_known INTEGER NOT NULL DEFAULT 1;
UPDATE jobs SET work_auth_known = 0 WHERE TRIM(description) = '' AND work_auth_flag = 0;
CREATE TABLE detail_queue_cursors (
    source TEXT PRIMARY KEY,
    cursor TEXT NOT NULL
);
