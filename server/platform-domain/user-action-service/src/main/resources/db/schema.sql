-- User Action Service Schema
-- Tables for product collection, read history, brand attention, and product comment

CREATE TABLE IF NOT EXISTS tb_product_collection (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(255),
    product_pic VARCHAR(500),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_product_collection_user_id ON tb_product_collection(user_id);
CREATE INDEX IF NOT EXISTS idx_product_collection_product_id ON tb_product_collection(product_id);

CREATE TABLE IF NOT EXISTS tb_read_history (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(255),
    product_pic VARCHAR(500),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_read_history_user_id ON tb_read_history(user_id);
CREATE INDEX IF NOT EXISTS idx_read_history_product_id ON tb_read_history(product_id);

CREATE TABLE IF NOT EXISTS tb_brand_attention (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT NOT NULL,
    brand_id BIGINT NOT NULL,
    brand_name VARCHAR(100),
    brand_logo VARCHAR(500),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_brand_attention_user_id ON tb_brand_attention(user_id);
CREATE INDEX IF NOT EXISTS idx_brand_attention_brand_id ON tb_brand_attention(brand_id);

CREATE TABLE IF NOT EXISTS tb_product_comment (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    product_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    member_nick_name VARCHAR(64),
    product_name VARCHAR(200),
    star INT DEFAULT 5,
    content TEXT,
    pics VARCHAR(1000),
    product_attribute VARCHAR(500),
    show_status INT DEFAULT 1,
    collect_count INT DEFAULT 0,
    read_count INT DEFAULT 0,
    replay_count INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_product_comment_product_id ON tb_product_comment(product_id);
CREATE INDEX IF NOT EXISTS idx_product_comment_user_id ON tb_product_comment(user_id);
