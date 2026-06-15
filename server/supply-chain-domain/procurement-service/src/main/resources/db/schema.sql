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