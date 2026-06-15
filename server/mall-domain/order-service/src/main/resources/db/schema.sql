-- Order Service Schema (OMS);

CREATE TABLE IF NOT EXISTS tb_order (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    coupon_id BIGINT,
    order_sn VARCHAR(64) NOT NULL UNIQUE,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    member_username VARCHAR(64),
    total_amount DECIMAL(18, 2) DEFAULT 0.00,
    pay_amount DECIMAL(18, 2) DEFAULT 0.00,
    freight_amount DECIMAL(18, 2) DEFAULT 0.00,
    promotion_amount DECIMAL(18, 2) DEFAULT 0.00,
    integration_amount DECIMAL(18, 2) DEFAULT 0.00,
    coupon_amount DECIMAL(18, 2) DEFAULT 0.00,
    discount_amount DECIMAL(18, 2) DEFAULT 0.00,
    pay_type INT,
    source_type INT,
    status INT,
    order_type INT,
    delivery_company VARCHAR(64),
    delivery_sn VARCHAR(64),
    auto_confirm_day INT,
    integration INT,
    growth INT,
    promotion_info VARCHAR(255),
    bill_type INT,
    bill_header VARCHAR(200),
    bill_content VARCHAR(200),
    bill_receiver_phone VARCHAR(32),
    bill_receiver_email VARCHAR(64),
    receiver_name VARCHAR(100) NOT NULL,
    receiver_phone VARCHAR(32) NOT NULL,
    receiver_post_code VARCHAR(32),
    receiver_province VARCHAR(32),
    receiver_city VARCHAR(32),
    receiver_region VARCHAR(32),
    receiver_detail_address VARCHAR(200),
    note VARCHAR(500),
    confirm_status SMALLINT,
    delete_status SMALLINT DEFAULT 0,
    member_receive_address_id BIGINT,
    use_integration INT,
    payment_time TIMESTAMP,
    delivery_time TIMESTAMP,
    receive_time TIMESTAMP,
    comment_time TIMESTAMP,
    modify_time TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_order_item (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    order_sn VARCHAR(64) NOT NULL,
    product_id BIGINT NOT NULL,
    product_pic VARCHAR(255),
    product_name VARCHAR(200),
    product_brand VARCHAR(200),
    product_sn VARCHAR(64),
    product_price DECIMAL(10, 2),
    product_quantity INT,
    product_sku_id BIGINT,
    product_sku_code VARCHAR(50),
    product_category_id BIGINT,
    promotion_name VARCHAR(255),
    promotion_amount DECIMAL(10, 2),
    coupon_amount DECIMAL(10, 2),
    integration_amount DECIMAL(10, 2),
    real_amount DECIMAL(10, 2),
    gift_integration INT DEFAULT 0,
    gift_growth INT DEFAULT 0,
    product_attr VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS tb_order_return_apply (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    company_address_id BIGINT,
    product_id BIGINT,
    order_sn VARCHAR(64) NOT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    member_username VARCHAR(64),
    return_amount DECIMAL(10, 2),
    return_name VARCHAR(100),
    return_phone VARCHAR(100),
    status INT DEFAULT 0,
    handle_time TIMESTAMP,
    product_pic VARCHAR(500),
    product_name VARCHAR(200),
    product_brand VARCHAR(200),
    product_attr VARCHAR(500),
    product_count INT,
    product_price DECIMAL(10, 2),
    product_real_price DECIMAL(10, 2),
    reason VARCHAR(200),
    description VARCHAR(500),
    proof_pics VARCHAR(1000),
    handle_note VARCHAR(500),
    handle_man VARCHAR(100),
    receive_man VARCHAR(100),
    receive_time TIMESTAMP,
    receive_note VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS tb_order_return_reason (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    sort INT DEFAULT 0,
    status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_order_setting (
    id BIGINT PRIMARY KEY,
    flash_order_overtime INT,
    normal_order_overtime INT,
    confirm_overtime INT,
    finish_overtime INT,
    comment_overtime INT
);

CREATE TABLE IF NOT EXISTS tb_company_address (
    id BIGINT PRIMARY KEY,
    address_name VARCHAR(200),
    send_status SMALLINT DEFAULT 0,
    receive_status SMALLINT DEFAULT 0,
    name VARCHAR(64),
    phone VARCHAR(32),
    province VARCHAR(64),
    city VARCHAR(64),
    region VARCHAR(64),
    detail_address VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS tb_order_operate_log (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    operate_man VARCHAR(100),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_status INT,
    note VARCHAR(500)
);

-- Saga Transaction Instance Table
CREATE TABLE IF NOT EXISTS tb_saga_instance (
    id BIGSERIAL PRIMARY KEY,
    saga_id VARCHAR(64) NOT NULL,  -- Saga transaction ID,
    biz_id VARCHAR(64),  -- Business ID (e.g., order ID),
    saga_type VARCHAR(32) NOT NULL,  -- Saga type: ORDER_PAYMENT_SAGA,
    status VARCHAR(32) NOT NULL DEFAULT 'RUNNING',  -- Status: RUNNING/COMPLETED/COMPENSATED/FAILED,
    current_step VARCHAR(32),  -- Current executing step,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Start time,
    end_time TIMESTAMP,  -- End time,
    error_message VARCHAR(1024),  -- Error message if failed,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_saga_id UNIQUE (saga_id)
);
CREATE INDEX IF NOT EXISTS idx_biz_id ON tb_saga_instance (biz_id);
CREATE INDEX IF NOT EXISTS idx_status ON tb_saga_instance (status);
CREATE INDEX IF NOT EXISTS idx_saga_type ON tb_saga_instance (saga_type);

-- Saga Step Execution Table
CREATE TABLE IF NOT EXISTS tb_saga_step (
    id BIGSERIAL PRIMARY KEY,
    saga_id VARCHAR(64) NOT NULL,  -- Saga transaction ID,
    step_name VARCHAR(64) NOT NULL,  -- Step name,
    step_order INT NOT NULL,  -- Step execution order,
    service_name VARCHAR(64) NOT NULL,  -- Target service name,
    compensable BOOLEAN DEFAULT TRUE,  -- Whether this step can be compensated,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',  -- Status: PENDING/RUNNING/COMPLETED/COMPENSATED/FAILED,
    request_data TEXT,  -- Request data JSON,
    response_data TEXT,  -- Response data JSON,
    compensation_data TEXT,  -- Compensation data JSON,
    retry_count INT DEFAULT 0,  -- Retry count,
    max_retries INT DEFAULT 3,  -- Max retries,
    error_message VARCHAR(1024),  -- Error message if failed,
    start_time TIMESTAMP,  -- Step start time,
    end_time TIMESTAMP,  -- Step end time,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_saga_id ON tb_saga_step (saga_id);
CREATE INDEX IF NOT EXISTS idx_status ON tb_saga_step (status);
CREATE INDEX IF NOT EXISTS idx_step_order ON tb_saga_step (step_order);
