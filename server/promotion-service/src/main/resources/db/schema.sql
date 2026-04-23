-- Promotion Service Schema (SMS)
-- Aligned with mall-admin-web and mall-app-web requirements

CREATE TABLE IF NOT EXISTS tb_coupon (
    id BIGINT PRIMARY KEY,
    type INT DEFAULT 0, -- 0->all, 1->member, 2->buy, 3->register
    name VARCHAR(100) NOT NULL,
    platform INT DEFAULT 0, -- 0->all, 1->mobile, 2->pc
    count INT DEFAULT 0,
    amount DECIMAL(10, 2) DEFAULT 0.00,
    per_limit INT DEFAULT 1,
    min_point DECIMAL(10, 2) DEFAULT 0.00,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    use_type INT DEFAULT 0, -- 0->all, 1->category, 2->product
    note VARCHAR(255),
    publish_count INT DEFAULT 0,
    use_count INT DEFAULT 0,
    receive_count INT DEFAULT 0,
    enable_time TIMESTAMP,
    code VARCHAR(64),
    member_level INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_coupon_history (
    id BIGINT PRIMARY KEY,
    coupon_id BIGINT NOT NULL,
    member_id BIGINT NOT NULL,
    coupon_code VARCHAR(64),
    member_nickname VARCHAR(64),
    get_type INT DEFAULT 0, -- 0->direct, 1->system
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    use_status INT DEFAULT 0, -- 0->unused, 1->used, 2->expired
    use_time TIMESTAMP,
    order_id BIGINT,
    order_sn VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS tb_coupon_product_relation (
    id BIGINT PRIMARY KEY,
    coupon_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    product_sn VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS tb_coupon_product_category_relation (
    id BIGINT PRIMARY KEY,
    coupon_id BIGINT NOT NULL,
    product_category_id BIGINT NOT NULL,
    product_category_name VARCHAR(200),
    parent_category_name VARCHAR(200)
);

CREATE TABLE IF NOT EXISTS tb_flash_promotion (
    id BIGINT PRIMARY KEY,
    title VARCHAR(200),
    start_date DATE,
    end_date DATE,
    status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_flash_promotion_session (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    start_time TIME,
    end_time TIME,
    status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_flash_promotion_product_relation (
    id BIGINT PRIMARY KEY,
    flash_promotion_id BIGINT NOT NULL,
    flash_promotion_session_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    flash_promotion_price DECIMAL(10, 2),
    flash_promotion_count INT,
    flash_promotion_limit INT,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_flash_promotion_log (
    id BIGINT PRIMARY KEY,
    member_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    member_phone VARCHAR(32),
    product_name VARCHAR(200),
    subscribe_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    send_time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tb_home_advertise (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100),
    type INT DEFAULT 0, -- 0->pc, 1->app
    pic VARCHAR(255),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status SMALLINT DEFAULT 1,
    click_count INT DEFAULT 0,
    order_count INT DEFAULT 0,
    url VARCHAR(500),
    note VARCHAR(255),
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_brand (
    id BIGINT PRIMARY KEY,
    brand_id BIGINT NOT NULL,
    brand_name VARCHAR(100),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_new_product (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_recommend_product (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tb_home_recommend_subject (
    id BIGINT PRIMARY KEY,
    subject_id BIGINT NOT NULL,
    subject_name VARCHAR(200),
    recommend_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0
);
