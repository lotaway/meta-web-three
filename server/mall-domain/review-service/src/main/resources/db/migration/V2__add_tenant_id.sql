ALTER TABLE review ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_review_tenant ON review(tenant_id);
