-- Product category table
CREATE TABLE tb_product_category (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT DEFAULT 0,
    name VARCHAR(100) NOT NULL,
    level INT DEFAULT 0,
    product_count INT DEFAULT 0,
    product_unit VARCHAR(50),
    nav_status SMALLINT DEFAULT 0,
    show_status SMALLINT DEFAULT 1,
    sort INT DEFAULT 0,
    icon VARCHAR(255),
    keywords VARCHAR(255),
    description TEXT
);

-- Brand table
CREATE TABLE tb_brand (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    first_letter VARCHAR(1) DEFAULT '',
    sort INT DEFAULT 0,
    factory_status SMALLINT DEFAULT 0,
    show_status SMALLINT DEFAULT 1,
    product_count INT DEFAULT 0,
    product_comment_count INT DEFAULT 0,
    logo VARCHAR(255),
    big_pic VARCHAR(255),
    brand_story TEXT
);

-- Product attribute category table
CREATE TABLE tb_product_attribute_category (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    attribute_count INT DEFAULT 0,
    param_count INT DEFAULT 0
);

-- Product attribute table
CREATE TABLE tb_product_attribute (
    id BIGSERIAL PRIMARY KEY,
    product_attribute_category_id BIGINT NOT NULL,
    name VARCHAR(100) NOT NULL,
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

-- Product table
CREATE TABLE tb_product (
    id BIGSERIAL PRIMARY KEY,
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

-- SKU Stock table
CREATE TABLE tb_sku_stock (
    id BIGSERIAL PRIMARY KEY,
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

-- Product attribute value table
CREATE TABLE tb_product_attribute_value (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_attribute_id BIGINT NOT NULL,
    value VARCHAR(255)
);

-- Member price table
CREATE TABLE tb_member_price (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    member_level_id BIGINT NOT NULL,
    member_price DECIMAL(10, 2) DEFAULT 0.00,
    member_level_name VARCHAR(100)
);

-- Product ladder table
CREATE TABLE tb_product_ladder (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    count INT DEFAULT 0,
    discount DECIMAL(10, 2) DEFAULT 0.00,
    price DECIMAL(10, 2) DEFAULT 0.00
);

-- Product full reduction table
CREATE TABLE tb_product_full_reduction (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL,
    full_price DECIMAL(10, 2) DEFAULT 0.00,
    reduce_price DECIMAL(10, 2) DEFAULT 0.00
);

-- Product Category Attribute Relation
CREATE TABLE tb_product_category_attribute_relation (
    id BIGSERIAL PRIMARY KEY,
    product_category_id BIGINT NOT NULL,
    product_attribute_id BIGINT NOT NULL
);

-- Freight template
CREATE TABLE tb_feight_template (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    charge_type INT DEFAULT 0,
    first_weight DECIMAL(10, 2),
    first_fee DECIMAL(10, 2),
    continue_weight DECIMAL(10, 2),
    continue_fee DECIMAL(10, 2),
    dest VARCHAR(255)
);
