ALTER TABLE after_sale_order ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_after_sale_order_tenant ON after_sale_order(tenant_id);
