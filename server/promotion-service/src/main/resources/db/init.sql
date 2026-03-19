CREATE TABLE coupon_type (
    id BIGINT PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    description VARCHAR(255),
    image_url VARCHAR(255),
    minimum_order_amount DECIMAL(18, 2) NOT NULL,
    discount_amount DECIMAL(18, 2) NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    create_user_id BIGINT,
    type_code VARCHAR(64),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_coupon_type_time ON coupon_type(start_time, end_time);
CREATE INDEX idx_coupon_type_enabled ON coupon_type(is_enabled);

CREATE TABLE coupon (
    id BIGINT PRIMARY KEY,
    code VARCHAR(32) NOT NULL UNIQUE,
    coupon_type_id BIGINT NOT NULL,
    owner_user_id BIGINT NOT NULL DEFAULT 0,
    transfer_status SMALLINT NOT NULL DEFAULT 0,
    acquire_method SMALLINT NOT NULL DEFAULT 0,
    use_status SMALLINT NOT NULL DEFAULT 0,
    order_no VARCHAR(64),
    consumer_name VARCHAR(64),
    operator_name VARCHAR(64),
    batch_id VARCHAR(64),
    used_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_coupon_owner_status ON coupon(owner_user_id, use_status);
CREATE INDEX idx_coupon_type ON coupon(coupon_type_id);
CREATE INDEX idx_coupon_code ON coupon(code);
CREATE INDEX idx_coupon_batch ON coupon(batch_id);

CREATE TABLE coupon_batch (
    id VARCHAR(64) PRIMARY KEY,
    coupon_type_id BIGINT NOT NULL,
    total_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_coupon_batch_type ON coupon_batch(coupon_type_id);
