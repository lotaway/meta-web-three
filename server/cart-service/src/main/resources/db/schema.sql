-- Cart service schema

CREATE TABLE IF NOT EXISTS oms_cart_item (
    id BIGINT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    product_sku_id BIGINT,
    member_id BIGINT NOT NULL,
    quantity INT NOT NULL DEFAULT 1,
    price DECIMAL(10, 2) DEFAULT 0.00,
    product_pic VARCHAR(255),
    product_name VARCHAR(200),
    product_sub_title VARCHAR(500),
    product_sku_code VARCHAR(50),
    member_nickname VARCHAR(100),
    create_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modify_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delete_status INT DEFAULT 0,
    product_category_id BIGINT,
    product_brand VARCHAR(200),
    product_sn VARCHAR(64),
    product_attr VARCHAR(500)
);

CREATE INDEX IF NOT EXISTS idx_cart_member_id ON oms_cart_item (member_id);
CREATE INDEX IF NOT EXISTS idx_cart_product_id ON oms_cart_item (product_id);
CREATE INDEX IF NOT EXISTS idx_cart_product_sku_id ON oms_cart_item (product_sku_id);
