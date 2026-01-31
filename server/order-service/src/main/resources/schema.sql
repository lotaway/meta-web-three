-- SQL Schema for Order Service (based on shopbest order module)

-- Order main table
CREATE TABLE tb_order (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    order_no VARCHAR(64) NOT NULL,
    order_status VARCHAR(20) NOT NULL, -- e.g., CREATED, PAID, CANCELED, SHIPPED, COMPLETED
    order_type VARCHAR(20) DEFAULT 'NORMAL',
    order_amount DECIMAL(18, 2) NOT NULL,
    order_remark TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Order item (SKU) table
CREATE TABLE tb_order_item (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    product_name VARCHAR(200),
    sku_id BIGINT,
    quantity INT NOT NULL,
    unit_price DECIMAL(18, 2) NOT NULL,
    total_price DECIMAL(18, 2) NOT NULL,
    image_url VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_order_item_order FOREIGN KEY (order_id) REFERENCES tb_order (id)
);

