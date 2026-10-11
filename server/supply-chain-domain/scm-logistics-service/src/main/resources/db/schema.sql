-- logistics-service

CREATE TABLE IF NOT EXISTS logistics_carrier (
    id BIGSERIAL PRIMARY KEY,
    carrier_code VARCHAR(64) NOT NULL UNIQUE,
    carrier_name VARCHAR(128) NOT NULL,
 carrier_type VARCHAR(32), -- EXPRESS/FREIGHT/POST,
    contact VARCHAR(64),
    phone VARCHAR(32),
    website VARCHAR(256),
 status VARCHAR(32) DEFAULT 'ACTIVE', -- ACTIVE/INACTIVE,
    base_freight DECIMAL(10,2) DEFAULT 0,
 weight_unit_price DECIMAL(10,4) DEFAULT 0, -- (/),
 volume_unit_price DECIMAL(10,4) DEFAULT 0, -- (/),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_carrier_code ON logistics_carrier (carrier_code);
CREATE INDEX IF NOT EXISTS idx_status ON logistics_carrier (status);

CREATE TABLE IF NOT EXISTS logistics_order (
    id BIGSERIAL PRIMARY KEY,
    tracking_no VARCHAR(64) NOT NULL UNIQUE,
    order_no VARCHAR(64) NOT NULL,
 carrier_id BIGINT, -- ID,
    carrier_name VARCHAR(128),
    service_type VARCHAR(32),
    sender_name VARCHAR(64) NOT NULL,
    sender_phone VARCHAR(32) NOT NULL,
    sender_province VARCHAR(32),
    sender_city VARCHAR(32),
    sender_district VARCHAR(32),
    sender_address VARCHAR(256),
    receiver_name VARCHAR(64) NOT NULL,
    receiver_phone VARCHAR(32) NOT NULL,
    receiver_province VARCHAR(32),
    receiver_city VARCHAR(32),
    receiver_district VARCHAR(32),
    receiver_address VARCHAR(256),
 weight DECIMAL(10,3), -- (kg),
    volume DECIMAL(10,3),
    freight DECIMAL(10,2) DEFAULT 0,
 status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/PICKED_UP/IN_TRANSIT/OUT_FOR_DELIVERY/DELIVERED/EXCEPTION,
    picked_up_at TIMESTAMP,
    in_transit_at TIMESTAMP,
    out_for_delivery_at TIMESTAMP,
    delivered_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tracking_no ON logistics_order (tracking_no);
CREATE INDEX IF NOT EXISTS idx_order_no ON logistics_order (order_no);
CREATE INDEX IF NOT EXISTS idx_carrier_id ON logistics_order (carrier_id);
CREATE INDEX IF NOT EXISTS idx_status ON logistics_order (status);

CREATE TABLE IF NOT EXISTS logistics_tracking_event (
    id BIGSERIAL PRIMARY KEY,
    tracking_no VARCHAR(64) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    location VARCHAR(128),
    description VARCHAR(512),
    operator VARCHAR(64),
    occurred_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tracking_no ON logistics_tracking_event (tracking_no);
CREATE INDEX IF NOT EXISTS idx_occurred_at ON logistics_tracking_event (occurred_at);

-- ---------- Part from procurement-service ----------

-- schema.sql

-- =====================
-- V1: procurement init
-- =====================

CREATE TABLE IF NOT EXISTS procurement_order (
    id BIGSERIAL PRIMARY KEY,
    order_no VARCHAR(64) NOT NULL UNIQUE,
    supplier_code VARCHAR(64),
    warehouse_id BIGINT,
    purchase_type VARCHAR(32), -- STOCK/PRODUCTION,
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PENDING/APPROVED/REJECTED/COMPLETED/CANCELLED,
    total_amount DECIMAL(15,2) DEFAULT 0,
    currency VARCHAR(8) DEFAULT 'CNY',
    payment_terms VARCHAR(64), -- NET_30/NET_60/NET_90,
    delivery_terms VARCHAR(64), -- FOB/CIF/EXW,
    remark TEXT,
    approver VARCHAR(128),
    approved_at TIMESTAMP,
    expected_delivery_date TIMESTAMP,
    actual_delivery_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX idx_procurement_order_no ON procurement_order(order_no);
CREATE INDEX idx_procurement_status ON procurement_order(status);
CREATE INDEX idx_procurement_supplier ON procurement_order(supplier_code);
CREATE INDEX idx_procurement_warehouse ON procurement_order(warehouse_id);

-- ---------- Part from rma-service ----------

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

