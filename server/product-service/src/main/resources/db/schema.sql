-- Product Service Schema (PMS + CMS relevant)
-- Aligned with mall-admin-web and mall-app-web requirements

CREATE TABLE tb_product_category (
    id BIGINT PRIMARY KEY,
    parent_id BIGINT DEFAULT 0,
    name VARCHAR(64) NOT NULL,
    level INT DEFAULT 0,
    product_count INT DEFAULT 0,
    product_unit VARCHAR(64),
    nav_status SMALLINT DEFAULT 0,
    show_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0,
    icon VARCHAR(255),
    keywords VARCHAR(255),
    description TEXT
);

CREATE TABLE tb_brand (
    id BIGINT PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    first_letter VARCHAR(8),
    sort INT DEFAULT 0,
    factory_status SMALLINT DEFAULT 0,
    show_status SMALLINT DEFAULT 1,
    product_count INT DEFAULT 0,
    product_comment_count INT DEFAULT 0,
    logo VARCHAR(255),
    big_pic VARCHAR(255),
    brand_story TEXT
);

CREATE TABLE tb_product_attribute_category (
    id BIGINT PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    attribute_count INT DEFAULT 0,
    param_count INT DEFAULT 0
);

CREATE TABLE tb_product_attribute (
    id BIGINT PRIMARY KEY,
    product_attribute_category_id BIGINT NOT NULL,
    name VARCHAR(64) NOT NULL,
    select_type INT DEFAULT 0,
    input_type INT DEFAULT 0,
    input_list TEXT,
    sort INT DEFAULT 0,
    filter_type INT DEFAULT 0,
    search_type INT DEFAULT 0,
    related_status SMALLINT DEFAULT 0,
    hand_add_status SMALLINT DEFAULT 0,
    type INT DEFAULT 0
);

CREATE TABLE tb_product (
    id BIGINT PRIMARY KEY,
    brand_id BIGINT,
    product_category_id BIGINT,
    feight_template_id BIGINT,
    product_attribute_category_id BIGINT,
    name VARCHAR(200) NOT NULL,
    pic VARCHAR(255),
    product_sn VARCHAR(64) NOT NULL,
    delete_status SMALLINT DEFAULT 0,
    publish_status SMALLINT DEFAULT 1,
    new_status SMALLINT DEFAULT 0,
    recommand_status SMALLINT DEFAULT 0,
    verify_status SMALLINT DEFAULT 0,
    sort INT DEFAULT 0,
    sale INT DEFAULT 0,
    price DECIMAL(10, 2) DEFAULT 0.00,
    promotion_price DECIMAL(10, 2),
    gift_growth INT DEFAULT 0,
    gift_point INT DEFAULT 0,
    use_point_limit INT,
    sub_title VARCHAR(255),
    description TEXT,
    original_price DECIMAL(10, 2),
    stock INT DEFAULT 0,
    low_stock INT DEFAULT 0,
    unit VARCHAR(16),
    weight DECIMAL(10, 2),
    preview_status SMALLINT DEFAULT 0,
    service_ids VARCHAR(64),
    keywords VARCHAR(255),
    note VARCHAR(255),
    album_pics VARCHAR(255),
    detail_title VARCHAR(255),
    detail_desc TEXT,
    detail_html TEXT,
    detail_mobile_html TEXT,
    promotion_start_time TIMESTAMP,
    promotion_end_time TIMESTAMP,
    promotion_per_limit INT,
    promotion_type INT DEFAULT 0,
    brand_name VARCHAR(255),
    product_category_name VARCHAR(255)
);

CREATE TABLE tb_sku_stock (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    sku_code VARCHAR(64) NOT NULL,
    price DECIMAL(10, 2) DEFAULT 0.00,
    stock INT DEFAULT 0,
    low_stock INT DEFAULT 0,
    pic VARCHAR(255),
    sale INT DEFAULT 0,
    promotion_price DECIMAL(10, 2),
    lock_stock INT DEFAULT 0,
    sp_data TEXT
);

CREATE TABLE tb_product_attribute_value (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_attribute_id BIGINT NOT NULL,
    value VARCHAR(255)
);

CREATE TABLE tb_product_category_attribute_relation (
    id BIGINT PRIMARY KEY,
    product_category_id BIGINT NOT NULL,
    product_attribute_id BIGINT NOT NULL
);

-- CMS Related (Subjects, etc.)
CREATE TABLE tb_subject (
    id BIGINT PRIMARY KEY,
    category_id BIGINT,
    title VARCHAR(100),
    pic VARCHAR(255),
    product_count INT DEFAULT 0,
    recommend_status SMALLINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    collect_count INT DEFAULT 0,
    read_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    album_pics VARCHAR(255),
    description VARCHAR(255),
    show_status SMALLINT DEFAULT 1,
    content TEXT,
    forward_count INT DEFAULT 0,
    category_name VARCHAR(64)
);

CREATE TABLE tb_subject_product_relation (
    id BIGINT PRIMARY KEY,
    subject_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL
);

CREATE TABLE tb_prefrence_area (
    id BIGINT PRIMARY KEY,
    name VARCHAR(64),
    sub_title VARCHAR(64),
    pic VARCHAR(255),
    sort INT DEFAULT 0,
    show_status SMALLINT DEFAULT 1
);

CREATE TABLE tb_prefrence_area_product_relation (
    id BIGINT PRIMARY KEY,
    prefrence_area_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL
);
