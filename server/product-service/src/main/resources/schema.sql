CREATE TABLE tb_category (
    id VARCHAR(36) PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    category_identity INT NOT NULL,
    parent_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000000',
    parent_id_str TEXT,
    sort_order INT DEFAULT 0
);

CREATE TABLE tb_brand (
    id BIGSERIAL PRIMARY KEY,
    brand_name VARCHAR(100) NOT NULL,
    language_version VARCHAR(10) DEFAULT 'zh-CN'
);

CREATE TABLE tb_product (
    id BIGSERIAL PRIMARY KEY,
    product_no VARCHAR(50),
    product_name VARCHAR(200) NOT NULL,
    creator INT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_remark TEXT,
    is_shelves SMALLINT DEFAULT 1,
    language_version VARCHAR(10) DEFAULT 'zh-CN'
);

CREATE TABLE tb_product_marketing (
    product_id BIGINT PRIMARY KEY,
    is_bargain SMALLINT DEFAULT 0,
    is_discount SMALLINT DEFAULT 0,
    is_total_reduce SMALLINT DEFAULT 0,
    is_gifts SMALLINT DEFAULT 0,
    is_send_integral SMALLINT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_shipping (
    product_id BIGINT PRIMARY KEY,
    is_from_freight SMALLINT DEFAULT 0,
    freight_template INT,
    is_order_from_freight SMALLINT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_limits (
    product_id BIGINT PRIMARY KEY,
    purchase INT DEFAULT 0,
    purchase_times INT DEFAULT 0,
    purchase_unit VARCHAR(20),
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_aftersales (
    product_id BIGINT PRIMARY KEY,
    is_refund_after_sd SMALLINT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_stats (
    product_id BIGINT PRIMARY KEY,
    comment_number INT DEFAULT 0,
    score_number INT DEFAULT 0,
    scores DECIMAL(3, 1) DEFAULT 0.0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_entity (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_artno VARCHAR(50),
    sale_price DECIMAL(18, 2) NOT NULL,
    market_price DECIMAL(18, 2) DEFAULT 0.0,
    inventory INT DEFAULT 0,
    image_url VARCHAR(255),
    is_user_discount SMALLINT DEFAULT 0,
    cash_back DECIMAL(18, 2) DEFAULT 0.0,
    cash_back_cycle INT DEFAULT 0,
    cycle_unit INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_gallery (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    image_url VARCHAR(255),
    sort_order INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_category_mapping (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    category_id INT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_specifications_value (
    id BIGSERIAL PRIMARY KEY,
    specifications INT,
    specifications_value_name VARCHAR(50)
);

CREATE TABLE tb_product_attribute (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    attribute_name VARCHAR(100),
    attribute_input_type INT,
    attribute_value VARCHAR(255),
    input_value TEXT,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

CREATE TABLE tb_product_comment (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    user_name VARCHAR(100),
    contact VARCHAR(100),
    content TEXT,
    comment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    comment_ip VARCHAR(50),
    user_face VARCHAR(255),
    nick_name VARCHAR(100),
    img_src VARCHAR(255),
    is_show SMALLINT DEFAULT 1,
    order_no INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);
