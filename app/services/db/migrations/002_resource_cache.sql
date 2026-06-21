-- Migration 002: resource cache table
-- Stores normalized text content fetched from external URLs.
-- last_fetched is stored as ISO 8601 UTC (TEXT) for portability.

CREATE TABLE IF NOT EXISTS resource_cache (
    url          TEXT    NOT NULL PRIMARY KEY,
    content      TEXT    NOT NULL,
    last_fetched TEXT    NOT NULL,
    ttl_seconds  INTEGER NOT NULL DEFAULT 21600
);

CREATE INDEX IF NOT EXISTS idx_resource_cache_last_fetched
    ON resource_cache (last_fetched);
