
-- SKU
CREATE TABLE IF NOT EXISTS quality_standard (
    id BIGSERIAL PRIMARY KEY,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
    product_name VARCHAR(255),
 inspection_type VARCHAR(32) NOT NULL, -- FULL-, SAMPLE-, AUTO-,
 inspection_level VARCHAR(32) NOT NULL, -- NORMAL-, STRICT-,
 sample_rate DECIMAL(5,2) DEFAULT 100.00, -- (%),
 check_items TEXT, -- JSON,
    acceptance_qty INTEGER DEFAULT 0,
    defect_qty_threshold INTEGER DEFAULT 0,
 weight_tolerance DECIMAL(5,2) DEFAULT 0.00, -- (%),
    dimension_tolerance VARCHAR(64),
    packaging_requirement VARCHAR(512),
    label_requirement VARCHAR(512),
    is_active SMALLINT DEFAULT 1,
    remark VARCHAR(512),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0,
    CONSTRAINT uk_sku UNIQUE (sku_code, deleted)
);

CREATE TABLE IF NOT EXISTS quality_inspection (
    id BIGSERIAL PRIMARY KEY,
    inspection_no VARCHAR(64) NOT NULL,
 order_id BIGINT, -- ID,
    order_no VARCHAR(64),
    inbound_type VARCHAR(32),
 warehouse_id BIGINT, -- ID,
    supplier_code VARCHAR(64),
    supplier_name VARCHAR(255),
 inspection_type VARCHAR(32) NOT NULL, -- FULL-, SAMPLE-,
 inspection_status VARCHAR(32) NOT NULL, -- PENDING-, IN_PROGRESS-, PASSED-, FAILED-, CONCESSION-,
    total_quantity INTEGER DEFAULT 0,
    inspected_quantity INTEGER DEFAULT 0,
    qualified_quantity INTEGER DEFAULT 0,
    unqualified_quantity INTEGER DEFAULT 0,
    concession_quantity INTEGER DEFAULT 0,
 defect_rate DECIMAL(5,2) DEFAULT 0.00, -- (%),
    inspector VARCHAR(64),
    inspection_time TIMESTAMP,
    result_remark VARCHAR(512),
    is_auto_inspection SMALLINT DEFAULT 0,
    source_system VARCHAR(64),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0,
    CONSTRAINT uk_inspection_no UNIQUE (inspection_no, deleted)
);
CREATE TABLE IF NOT EXISTS quality_inspection_item (
    id BIGSERIAL PRIMARY KEY,
 inspection_id BIGINT NOT NULL, -- ID,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
    product_name VARCHAR(255),
    batch_no VARCHAR(64),
    location_code VARCHAR(64),
    plan_quantity INTEGER DEFAULT 0,
    actual_quantity INTEGER DEFAULT 0,
    inspected_quantity INTEGER DEFAULT 0,
    qualified_quantity INTEGER DEFAULT 0,
    unqualified_quantity INTEGER DEFAULT 0,
    concession_quantity INTEGER DEFAULT 0,
    sample_quantity INTEGER DEFAULT 0,
 defect_items TEXT, -- JSON,
 check_result VARCHAR(32) NOT NULL, -- QUALIFIED-, UNQUALIFIED-, CONCESSION-,
    remark VARCHAR(512),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inspection_id ON quality_inspection_item (inspection_id, deleted);
CREATE INDEX IF NOT EXISTS idx_sku ON quality_inspection_item (sku_code, deleted);

CREATE TABLE IF NOT EXISTS defect_record (
    id BIGSERIAL PRIMARY KEY,
 inspection_id BIGINT NOT NULL, -- ID,
 inspection_item_id BIGINT, -- ID,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
    product_name VARCHAR(255),
    batch_no VARCHAR(64),
    defect_type VARCHAR(64) NOT NULL,
    defect_name VARCHAR(128) NOT NULL,
    defect_description VARCHAR(512),
    defect_quantity INTEGER DEFAULT 0,
 defect_level VARCHAR(32) NOT NULL, -- CRITICAL-, MAJOR-, MINOR-,
 photo_urls TEXT, -- JSON,
    location_code VARCHAR(64),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inspection_id ON defect_record (inspection_id, deleted);
CREATE INDEX IF NOT EXISTS idx_sku ON defect_record (sku_code, deleted);

CREATE TABLE IF NOT EXISTS defect_processing (
    id BIGSERIAL PRIMARY KEY,
 defect_id BIGINT NOT NULL, -- ID,
    processing_no VARCHAR(64) NOT NULL,
 processing_type VARCHAR(32) NOT NULL, -- RETURN-, EXCHANGE-, DISCOUNT-, SCRAP-, SPECIAL_USE-,
 processing_status VARCHAR(32) NOT NULL, -- PENDING-, PROCESSING-, COMPLETED-, CANCELLED-,
    processing_quantity INTEGER DEFAULT 0,
 processing_price DECIMAL(12,2) DEFAULT 0.00, -- /,
    processing_reason VARCHAR(512),
    processing_remark VARCHAR(512),
    processor VARCHAR(64),
    processing_time TIMESTAMP,
    related_document_no VARCHAR(64),
    related_document_type VARCHAR(32),
    approver VARCHAR(64),
    approve_time TIMESTAMP,
    approve_remark VARCHAR(512),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0,
    CONSTRAINT uk_processing_no UNIQUE (processing_no, deleted)
);
CREATE INDEX IF NOT EXISTS idx_defect_id ON defect_processing (defect_id, deleted);

CREATE TABLE IF NOT EXISTS quality_config (
    id BIGSERIAL PRIMARY KEY,
    config_key VARCHAR(64) NOT NULL,
    config_value TEXT,
 config_type VARCHAR(32) NOT NULL, -- SYSTEM-, WAREHOUSE-,
 warehouse_id BIGINT, -- ID,
    description VARCHAR(255),
    is_active SMALLINT DEFAULT 1,
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0,
    CONSTRAINT uk_key_warehouse UNIQUE (config_key, warehouse_id, deleted)
);

CREATE INDEX idx_quality_standard_sku ON quality_standard(sku_code);
CREATE INDEX idx_quality_inspection_order ON quality_inspection(order_id);
CREATE INDEX idx_quality_inspection_status ON quality_inspection(inspection_status);
CREATE INDEX idx_defect_record_type ON defect_record(defect_type);
CREATE INDEX idx_defect_processing_status ON defect_processing(processing_status);
