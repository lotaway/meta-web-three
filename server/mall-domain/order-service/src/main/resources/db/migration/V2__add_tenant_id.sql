ALTER TABLE tb_order ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_order_tenant ON tb_order(tenant_id);

ALTER TABLE tb_order_item ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_order_item_tenant ON tb_order_item(tenant_id);

ALTER TABLE tb_order_return_apply ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_order_return_apply_tenant ON tb_order_return_apply(tenant_id);
