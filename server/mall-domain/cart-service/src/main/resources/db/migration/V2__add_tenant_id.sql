ALTER TABLE oms_cart_item ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_oms_cart_item_tenant ON oms_cart_item(tenant_id);
