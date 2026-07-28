ALTER TABLE tb_product ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_product_tenant ON tb_product(tenant_id);

ALTER TABLE tb_sku_stock ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_sku_stock_tenant ON tb_sku_stock(tenant_id);
