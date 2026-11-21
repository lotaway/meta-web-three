-- Add data-source provenance to devices so the frontend can distinguish
-- REAL (ingested from the field) from DEMO (seeded for demonstration).
ALTER TABLE devices ADD COLUMN IF NOT EXISTS source VARCHAR(10) NOT NULL DEFAULT 'REAL';

COMMENT ON COLUMN devices.source IS 'Data provenance: REAL or DEMO';
