-- schema.sql

-- =====================
-- V1: supplier init
-- =====================

CREATE TABLE IF NOT EXISTS supplier (
    id BIGSERIAL PRIMARY KEY,
    supplier_code VARCHAR(64) NOT NULL UNIQUE,
    supplier_name VARCHAR(128) NOT NULL,
    supplier_type VARCHAR(32), -- RAW_MATERIAL/PACKAGING/EQUIPMENT/SERVICES
    business_license VARCHAR(64),
    tax_id VARCHAR(64),
    province VARCHAR(64),
    city VARCHAR(64),
    district VARCHAR(64),
    address VARCHAR(512),
    contact VARCHAR(128),
    phone VARCHAR(32),
    email VARCHAR(128),
    status VARCHAR(32) DEFAULT 'ACTIVE', -- ACTIVE/INACTIVE/SUSPENDED
    credit_limit DECIMAL(15,2),
    payment_terms VARCHAR(64), -- NET_30/NET_60/NET_90
    category VARCHAR(64)
    score INTEGER DEFAULT 0,
    level VARCHAR(32), -- A/B/C/D
    assessment_level VARCHAR(32), -- EXCELLENT/GOOD/FAIR/POOR
    contact_person VARCHAR(128),
    contact_phone VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX idx_supplier_code ON supplier(supplier_code);
CREATE INDEX idx_supplier_status ON supplier(status);
CREATE INDEX idx_supplier_category ON supplier(category);
CREATE INDEX idx_supplier_assessment_level ON supplier(assessment_level);

-- =====================
-- V2: supplier portal init
-- =====================

CREATE TABLE IF NOT EXISTS supplier_shipment_notice (
    id BIGSERIAL PRIMARY KEY,
    notice_no VARCHAR(64) NOT NULL UNIQUE,
    supplier_code VARCHAR(64) NOT NULL,
    order_no VARCHAR(64) NOT NULL,
    warehouse_id BIGINT,
    expected_shipment_date TIMESTAMP,
    actual_shipment_date TIMESTAMP,
    shipment_method VARCHAR(32), -- EXPRESS/OCEAN/AIR/LAND
    carrier_name VARCHAR(128),
    carrier_contact VARCHAR(64),
    tracking_number VARCHAR(128),
    vehicle_number VARCHAR(64),
    driver_name VARCHAR(64),
    driver_phone VARCHAR(32),
    total_quantity DECIMAL(15,3),
    total_weight DECIMAL(15,3),
    total_volume DECIMAL(15,3),
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/SUBMITTED/CONFIRMED/IN_TRANSIT/DELIVERED/CANCELLED
    remark TEXT,
    confirmer VARCHAR(128),
    confirmed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX idx_shipment_notice_no ON supplier_shipment_notice(notice_no);
CREATE INDEX idx_shipment_supplier ON supplier_shipment_notice(supplier_code);
CREATE INDEX idx_shipment_order ON supplier_shipment_notice(order_no);
CREATE INDEX idx_shipment_status ON supplier_shipment_notice(status);

CREATE TABLE IF NOT EXISTS supplier_shipment_notice_item (
    id BIGSERIAL PRIMARY KEY,
    notice_id BIGINT NOT NULL,
    product_code VARCHAR(64) NOT NULL,
    product_name VARCHAR(128),
    unit VARCHAR(16),
    quantity DECIMAL(15,3),
    weight DECIMAL(15,3),
    volume DECIMAL(15,3),
    batch_no VARCHAR(64),
    production_date TIMESTAMP,
    expiry_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX idx_shipment_item_notice ON supplier_shipment_notice_item(notice_id);
CREATE INDEX idx_shipment_item_product ON supplier_shipment_notice_item(product_code);

CREATE TABLE IF NOT EXISTS supplier_reconciliation (
    id BIGSERIAL PRIMARY KEY,
    reconciliation_no VARCHAR(64) NOT NULL UNIQUE,
    supplier_code VARCHAR(64) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    order_count INTEGER DEFAULT 0,
    total_amount DECIMAL(15,2) DEFAULT 0,
    shipped_amount DECIMAL(15,2) DEFAULT 0,
    invoiced_amount DECIMAL(15,2) DEFAULT 0,
    settled_amount DECIMAL(15,2) DEFAULT 0,
    pending_amount DECIMAL(15,2) DEFAULT 0,
    currency VARCHAR(8) DEFAULT 'CNY',
    status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/SUBMITTED/CONFIRMED/REJECTED/PAID
    submitted_at TIMESTAMP,
    confirmed_at TIMESTAMP,
    confirmed_by VARCHAR(128),
    paid_at TIMESTAMP,
    remark TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX idx_reconciliation_no ON supplier_reconciliation(reconciliation_no);
CREATE INDEX idx_reconciliation_supplier ON supplier_reconciliation(supplier_code);
CREATE INDEX idx_reconciliation_status ON supplier_reconciliation(status);
CREATE INDEX idx_reconciliation_period ON supplier_reconciliation(period_start, period_end);

CREATE TABLE IF NOT EXISTS supplier_reconciliation_item (
    id BIGSERIAL PRIMARY KEY,
    reconciliation_id BIGINT NOT NULL,
    order_no VARCHAR(64) NOT NULL,
    order_date DATE,
    shipped_date DATE,
    invoiced_amount DECIMAL(15,2),
    settled_amount DECIMAL(15,2),
    pending_amount DECIMAL(15,2),
    status VARCHAR(32), -- PENDING/CONFIRMED/REJECTED
    remark TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    version INTEGER DEFAULT 0
);

CREATE INDEX idx_reconciliation_item_recon ON supplier_reconciliation_item(reconciliation_id);
CREATE INDEX idx_reconciliation_item_order ON supplier_reconciliation_item(order_no);

-- =====================
-- V3: supplier performance
-- =====================

CREATE TABLE IF NOT EXISTS supplier_performance (
    id BIGSERIAL PRIMARY KEY,
 supplier_id BIGINT NOT NULL, -- ID,
    supplier_code VARCHAR(50)
    supplier_name VARCHAR(200)
    period_start DATE NOT NULL
    period_end DATE NOT NULL
 on_time_delivery_rate DECIMAL(5, 2), -- (0-100),
 quality_pass_rate DECIMAL(5, 2), -- (0-100),
 price_competitiveness_score DECIMAL(5, 2), -- (0-100),
 overall_score DECIMAL(5, 2), -- (0-100),
 assessment_level VARCHAR(10), -- (A/B/C/D),
    total_orders INT
    on_time_delivery_count INT
    qualified_count INT
    total_quality_check_count INT
    market_avg_price DECIMAL(18, 2)
    supplier_price DECIMAL(18, 2)
    remark VARCHAR(500)
    assessor VARCHAR(100)
    assessment_date TIMESTAMP
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_supplier_performance_supplier_id ON supplier_performance (supplier_id);
CREATE INDEX IF NOT EXISTS idx_supplier_performance_period ON supplier_performance (period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_supplier_performance_assessment_level ON supplier_performance (assessment_level);