-- rma-service

CREATE TABLE IF NOT EXISTS rma_order (
    id BIGSERIAL PRIMARY KEY,
    rma_no VARCHAR(64) NOT NULL,
    order_no VARCHAR(64) NOT NULL,
    return_type VARCHAR(32) NOT NULL, -- REFUND/REPLACEMENT/REPAIR
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING', -- PENDING/AWAITING_INSPECTION/INSPECTED/AWAITING_DISPOSITION/DISPOSED/COMPLETED/CANCELLED
    customer_id BIGINT, -- ID
    customer_name VARCHAR(128),
    contact_phone VARCHAR(32),
    reason_code VARCHAR(64),
    reason_description VARCHAR(512),
    warehouse_id BIGINT, -- ID
    total_quantity INT DEFAULT 0,
    total_amount DECIMAL(20,2) DEFAULT 0.00,
    currency VARCHAR(8) DEFAULT 'CNY',
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0,
    CONSTRAINT uk_rma_no UNIQUE (rma_no)
);
CREATE INDEX IF NOT EXISTS idx_order_no ON rma_order (order_no);
CREATE INDEX IF NOT EXISTS idx_status ON rma_order (status);

CREATE TABLE IF NOT EXISTS rma_order_item (
    id BIGSERIAL PRIMARY KEY,
    rma_id BIGINT NOT NULL,
    sku_code VARCHAR(64) NOT NULL, -- SKU
    sku_name VARCHAR(256),
    expected_quantity INT DEFAULT 0,
    inspected_quantity INT DEFAULT 0,
    accepted_quantity INT DEFAULT 0,
    unit_price DECIMAL(20,2) DEFAULT 0.00,
    reason_code VARCHAR(64),
    reason_description VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rma_id ON rma_order_item (rma_id);
CREATE INDEX IF NOT EXISTS idx_sku_code ON rma_order_item (sku_code);

CREATE TABLE IF NOT EXISTS rma_inspection (
    id BIGSERIAL PRIMARY KEY,
    rma_id BIGINT NOT NULL,
    rma_no VARCHAR(64) NOT NULL,
    inspector VARCHAR(64),
    inspection_date TIMESTAMP,
    result VARCHAR(32), -- PASS/FAIL/PARTIAL
    conclusion VARCHAR(512),
    total_inspected INT DEFAULT 0,
    total_passed INT DEFAULT 0,
    total_failed INT DEFAULT 0,
    remark VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rma_id ON rma_inspection (rma_id);
CREATE INDEX IF NOT EXISTS idx_rma_no ON rma_inspection (rma_no);

CREATE TABLE IF NOT EXISTS rma_disposition (
    id BIGSERIAL PRIMARY KEY,
    rma_id BIGINT NOT NULL,
    rma_no VARCHAR(64) NOT NULL,
    disposition_type VARCHAR(32) NOT NULL, -- REFUND/REPLACEMENT/REPAIR/SCRAP/RETURN_TO_SUPPLIER
    refund_amount DECIMAL(20,2),
    replacement_sku_code VARCHAR(64),
    replacement_quantity INT,
    scrap_quantity INT,
    scrap_reason VARCHAR(512),
    disposition_by VARCHAR(64),
    disposition_date TIMESTAMP,
    remark VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rma_id ON rma_disposition (rma_id);
CREATE INDEX IF NOT EXISTS idx_rma_no ON rma_disposition (rma_no);

CREATE TABLE IF NOT EXISTS return_shipping (
    id BIGSERIAL PRIMARY KEY,
    rma_id BIGINT NOT NULL,
    rma_no VARCHAR(64) NOT NULL,
    carrier VARCHAR(64),
    tracking_no VARCHAR(128),
    shipping_method VARCHAR(32),
    origin_address VARCHAR(512),
    destination_address VARCHAR(512),
    shipping_date TIMESTAMP,
    estimated_arrival_date TIMESTAMP,
    status VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_tracking_no UNIQUE (tracking_no)
);
CREATE INDEX IF NOT EXISTS idx_rma_id ON return_shipping (rma_id);
CREATE INDEX IF NOT EXISTS idx_rma_no ON return_shipping (rma_no);
