-- schema.sql
-- 供应商服务数据库表结构

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
    category VARCHAR(64), -- 供应商分类
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