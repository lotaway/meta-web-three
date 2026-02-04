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
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_coupon_type_time (start_time, end_time),
    INDEX idx_coupon_type_enabled (is_enabled)
);

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
    used_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_coupon_owner_status (owner_user_id, use_status),
    INDEX idx_coupon_type (coupon_type_id),
    INDEX idx_coupon_code (code),
    INDEX idx_coupon_batch (batch_id)
);

CREATE TABLE coupon_batch (
    id VARCHAR(64) PRIMARY KEY,
    coupon_type_id BIGINT NOT NULL,
    total_count INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_coupon_batch_type (coupon_type_id)
);
