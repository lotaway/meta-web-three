-- Review Service Schema

CREATE TABLE IF NOT EXISTS review (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT,
    order_item_id BIGINT,
    product_id BIGINT,
    sku_id BIGINT,
    user_id BIGINT,
    store_id BIGINT,
    rating INT DEFAULT 5,
    content TEXT,
    images TEXT,
    status INT DEFAULT 0,
    like_count INT DEFAULT 0,
    reply_count INT DEFAULT 0,
    reply_content TEXT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tenant_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_review_tenant ON review(tenant_id);
CREATE INDEX IF NOT EXISTS idx_review_user_id ON review (user_id);
CREATE INDEX IF NOT EXISTS idx_review_product_id ON review (product_id);
CREATE INDEX IF NOT EXISTS idx_review_status ON review (status);