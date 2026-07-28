ALTER TABLE tb_coupon ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_coupon_tenant ON tb_coupon(tenant_id);

ALTER TABLE tb_coupon_history ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_coupon_history_tenant ON tb_coupon_history(tenant_id);

ALTER TABLE tb_flash_promotion ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_flash_promotion_tenant ON tb_flash_promotion(tenant_id);

ALTER TABLE tb_flash_promotion_session ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_flash_promotion_session_tenant ON tb_flash_promotion_session(tenant_id);

ALTER TABLE tb_flash_promotion_product_relation ADD COLUMN IF NOT EXISTS tenant_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_tb_flash_promotion_product_relation_tenant ON tb_flash_promotion_product_relation(tenant_id);
