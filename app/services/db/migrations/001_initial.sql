-- Migration 001: workflow history table
-- Uses IF NOT EXISTS so this script is safe to run multiple times.

CREATE TABLE IF NOT EXISTS workflow_runs (
    request_id      TEXT    PRIMARY KEY,
    user_input      TEXT    NOT NULL,
    intent          TEXT,
    selected_tool   TEXT,
    tool_output     TEXT,           -- JSON-encoded dict or null
    final_response  TEXT,
    error           TEXT,
    started_at      TEXT,
    completed_at    TEXT,
    created_at      TEXT    NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
