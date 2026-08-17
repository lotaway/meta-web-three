-- After-Sale Service Schema

CREATE TABLE IF NOT EXISTS after_sale_order (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL,
    order_no VARCHAR(64),
    user_id BIGINT,
    product_id BIGINT,
    sku_id BIGINT,
    product_name VARCHAR(200),
    product_image VARCHAR(500),
    quantity INT DEFAULT 1,
    refund_amount INT DEFAULT 0,
    after_sale_type INT DEFAULT 0,
    after_sale_status INT DEFAULT 0,
    apply_reason VARCHAR(500),
    reject_reason VARCHAR(500),
    apply_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    process_time TIMESTAMP,
    complete_time TIMESTAMP,
    remark VARCHAR(500),
    tenant_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_after_sale_order_tenant ON after_sale_order(tenant_id);
CREATE INDEX IF NOT EXISTS idx_after_sale_order_user_id ON after_sale_order (user_id);
CREATE INDEX IF NOT EXISTS idx_after_sale_order_status ON after_sale_order (after_sale_status);