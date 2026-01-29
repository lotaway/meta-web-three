-- SQL Schema for Product Service (Migrated from GoodsController.cs)

-- Category Table
CREATE TABLE tb_category (
    id VARCHAR(36) PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    category_identity INT NOT NULL,
    parent_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000000',
    parent_id_str TEXT,
    sort_order INT DEFAULT 0
);

-- Brand Table
CREATE TABLE tb_brand (
    id INT AUTO_INCREMENT PRIMARY KEY,
    brand_name VARCHAR(100) NOT NULL,
    language_version VARCHAR(10) DEFAULT 'zh-CN'
);

-- Product Table (Core identity and status)
CREATE TABLE tb_product (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_no VARCHAR(50),
    product_name VARCHAR(200) NOT NULL,
    creator INT,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    product_remark TEXT, -- Rich text description
    is_shelves TINYINT DEFAULT 1, -- Whether it's on the shelf
    language_version VARCHAR(10) DEFAULT 'zh-CN'
);

-- Product Marketing/Promotion Settings
CREATE TABLE tb_product_marketing (
    product_id INT PRIMARY KEY,
    is_bargain TINYINT DEFAULT 0,
    is_discount TINYINT DEFAULT 0,
    is_total_reduce TINYINT DEFAULT 0,
    is_gifts TINYINT DEFAULT 0,
    is_send_integral TINYINT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Shipping/Logistics Settings
CREATE TABLE tb_product_shipping (
    product_id INT PRIMARY KEY,
    is_from_freight TINYINT DEFAULT 0,
    freight_template INT,
    is_order_from_freight TINYINT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Purchase Limits
CREATE TABLE tb_product_limits (
    product_id INT PRIMARY KEY,
    purchase INT DEFAULT 0,
    purchase_times INT DEFAULT 0,
    purchase_unit VARCHAR(20),
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product After-sales Policy
CREATE TABLE tb_product_aftersales (
    product_id INT PRIMARY KEY,
    is_refund_after_sd TINYINT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Statistics and Ratings
CREATE TABLE tb_product_stats (
    product_id INT PRIMARY KEY,
    comment_number INT DEFAULT 0,
    score_number INT DEFAULT 0,
    scores DECIMAL(3, 1) DEFAULT 0.0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Entity Table (SKUs)
CREATE TABLE tb_product_entity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL, -- FK to tb_product
    product_artno VARCHAR(50),
    sale_price DECIMAL(18, 2) NOT NULL,
    market_price DECIMAL(18, 2) DEFAULT 0.0,
    inventory INT DEFAULT 0,
    image_url VARCHAR(255),
    is_user_discount TINYINT DEFAULT 0,
    cash_back DECIMAL(18, 2) DEFAULT 0.0,
    cash_back_cycle INT DEFAULT 0,
    cycle_unit INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Gallery (Images)
CREATE TABLE tb_product_gallery (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    image_url VARCHAR(255),
    sort_order INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Category Mapping
CREATE TABLE tb_product_category_mapping (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    category_id INT NOT NULL, -- Maps to category_identity
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Specifications Value Name Table
CREATE TABLE tb_specifications_value (
    id INT AUTO_INCREMENT PRIMARY KEY,
    specifications INT,
    specifications_value_name VARCHAR(50)
);

-- Product Attributes Table
CREATE TABLE tb_product_attribute (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    attribute_name VARCHAR(100),
    attribute_input_type INT, -- 3, 4 for multi-choice
    attribute_value VARCHAR(255), -- Comma separated IDs or direct value
    input_value TEXT,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);

-- Product Comments (@TODO: Change to message-service)
CREATE TABLE tb_product_comment (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    user_name VARCHAR(100),
    contact VARCHAR(100),
    content TEXT,
    comment_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    comment_ip VARCHAR(50),
    user_face VARCHAR(255),
    nick_name VARCHAR(100),
    img_src VARCHAR(255),
    is_show TINYINT DEFAULT 1,
    order_no INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES tb_product (id)
);