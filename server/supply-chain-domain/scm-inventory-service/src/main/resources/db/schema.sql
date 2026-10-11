-- inventory-service

CREATE TABLE IF NOT EXISTS inventory (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    total_quantity INT DEFAULT 0,
    available_quantity INT DEFAULT 0,
    reserved_quantity INT DEFAULT 0,
    defective_quantity INT DEFAULT 0,
    unit_cost DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0,
    CONSTRAINT uk_sku_warehouse UNIQUE (sku_code, warehouse_id)
);
CREATE INDEX IF NOT EXISTS idx_inventory_sku_code ON inventory (sku_code);
CREATE INDEX IF NOT EXISTS idx_inventory_warehouse_id ON inventory (warehouse_id);

CREATE TABLE IF NOT EXISTS inventory_record (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    biz_type VARCHAR(32) NOT NULL,
    biz_id VARCHAR(64),
    quantity INT NOT NULL,
    before_quantity INT,
    after_quantity INT,
    remark VARCHAR(512),
    operator VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_inventory_record_sku_code ON inventory_record (sku_code);
CREATE INDEX IF NOT EXISTS idx_inventory_record_warehouse_id ON inventory_record (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_inventory_record_biz_type ON inventory_record (biz_type);
CREATE INDEX IF NOT EXISTS idx_inventory_record_biz_id ON inventory_record (biz_id);
CREATE INDEX IF NOT EXISTS idx_inventory_record_created_at ON inventory_record (created_at);

CREATE TABLE IF NOT EXISTS replenishment_recommendation (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    current_stock INT,
    safety_stock INT,
    lead_time_days INT,
    average_daily_sales INT,
    recommended_quantity INT,
    recommendation_type VARCHAR(32) DEFAULT 'AUTO',
    status VARCHAR(32) DEFAULT 'PENDING',
    generated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_replenishment_sku_code ON replenishment_recommendation (sku_code);
CREATE INDEX IF NOT EXISTS idx_replenishment_warehouse_id ON replenishment_recommendation (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_replenishment_status ON replenishment_recommendation (status);
CREATE INDEX IF NOT EXISTS idx_replenishment_generated_at ON replenishment_recommendation (generated_at);

CREATE TABLE IF NOT EXISTS demand_forecast (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    forecast_period_days INT,
    predicted_quantity INT,
    confidence_level INT,
    forecast_method VARCHAR(32) DEFAULT 'SMA',
    forecast_start_date DATE,
    forecast_end_date DATE,
    status VARCHAR(32) DEFAULT 'PENDING',
    generated_at TIMESTAMP,
    notes VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_demand_forecast_sku_code ON demand_forecast (sku_code);
CREATE INDEX IF NOT EXISTS idx_demand_forecast_warehouse_id ON demand_forecast (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_demand_forecast_status ON demand_forecast (status);
CREATE INDEX IF NOT EXISTS idx_demand_forecast_dates ON demand_forecast (forecast_start_date, forecast_end_date);

CREATE TABLE IF NOT EXISTS inventory_batch (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    batch_no VARCHAR(64) NOT NULL,
    quantity INT NOT NULL,
    available_quantity INT DEFAULT 0,
    reserved_quantity INT DEFAULT 0,
    picked_quantity INT DEFAULT 0,
    inbound_date TIMESTAMP,
    production_date DATE,
    expiry_date DATE,
    unit_cost DECIMAL(10,2) DEFAULT 0,
    location_code VARCHAR(32),
    status VARCHAR(32) DEFAULT 'AVAILABLE',
    remark VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0,
    CONSTRAINT uk_batch UNIQUE (sku_code, warehouse_id, batch_no)
);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_sku_code ON inventory_batch (sku_code);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_warehouse_id ON inventory_batch (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_no ON inventory_batch (batch_no);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_inbound_date ON inventory_batch (inbound_date);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_expiry_date ON inventory_batch (expiry_date);
CREATE INDEX IF NOT EXISTS idx_inventory_batch_status ON inventory_batch (status);

CREATE TABLE IF NOT EXISTS outbound_strategy (
    id BIGSERIAL PRIMARY KEY,
    strategy_code VARCHAR(64) NOT NULL,
    strategy_name VARCHAR(128) NOT NULL,
    strategy_type VARCHAR(32) NOT NULL,
    warehouse_id BIGINT,
    warehouse_code VARCHAR(32),
    sku_code VARCHAR(64),
    sku_code_pattern VARCHAR(128),
    priority INT DEFAULT 100,
    specific_batch_no VARCHAR(64),
    is_active BOOLEAN DEFAULT TRUE,
    remark VARCHAR(512),
    creator VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0,
    CONSTRAINT uk_strategy_code UNIQUE (strategy_code)
);
CREATE INDEX IF NOT EXISTS idx_outbound_strategy_warehouse_id ON outbound_strategy (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_outbound_strategy_sku_code ON outbound_strategy (sku_code);
CREATE INDEX IF NOT EXISTS idx_outbound_strategy_type ON outbound_strategy (strategy_type);
CREATE INDEX IF NOT EXISTS idx_outbound_strategy_is_active ON outbound_strategy (is_active);
CREATE INDEX IF NOT EXISTS idx_outbound_strategy_priority ON outbound_strategy (priority);

CREATE TABLE IF NOT EXISTS inventory_alert_config (
    id BIGSERIAL PRIMARY KEY,
    config_code VARCHAR(64) NOT NULL,
    warehouse_code VARCHAR(32),
    sku_code VARCHAR(64),
    safety_stock_threshold INT,
    level VARCHAR(32) DEFAULT 'WARNING',
    enabled BOOLEAN DEFAULT TRUE,
    cooldown_minutes INT DEFAULT 60,
    notification_channels VARCHAR(128),
    notify_users VARCHAR(512),
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(64),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0,
    CONSTRAINT uk_config_code UNIQUE (config_code)
);
CREATE INDEX IF NOT EXISTS idx_alert_config_warehouse_code ON inventory_alert_config (warehouse_code);
CREATE INDEX IF NOT EXISTS idx_alert_config_sku_code ON inventory_alert_config (sku_code);
CREATE INDEX IF NOT EXISTS idx_alert_config_enabled ON inventory_alert_config (enabled);

CREATE TABLE IF NOT EXISTS inventory_alert (
    id BIGSERIAL PRIMARY KEY,
    alert_code VARCHAR(64) NOT NULL,
    warehouse_code VARCHAR(32) NOT NULL,
    sku_code VARCHAR(64) NOT NULL,
    alert_type VARCHAR(32) NOT NULL,
    level VARCHAR(32) DEFAULT 'WARNING',
    title VARCHAR(256) NOT NULL,
    description VARCHAR(1024),
    current_quantity INT,
    threshold_value INT,
    status VARCHAR(32) DEFAULT 'TRIGGERED',
    solution VARCHAR(1024),
    acknowledged_by VARCHAR(64),
    acknowledged_at TIMESTAMP,
    resolved_by VARCHAR(64),
    resolved_at TIMESTAMP,
    occurred_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0,
    CONSTRAINT uk_alert_code UNIQUE (alert_code)
);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_warehouse_code ON inventory_alert (warehouse_code);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_sku_code ON inventory_alert (sku_code);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_type ON inventory_alert (alert_type);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_level ON inventory_alert (level);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_status ON inventory_alert (status);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_occurred_at ON inventory_alert (occurred_at);
CREATE INDEX IF NOT EXISTS idx_inventory_alert_created_at ON inventory_alert (created_at);

CREATE TABLE IF NOT EXISTS inventory_reservation_record (
    id BIGSERIAL PRIMARY KEY,
    biz_id VARCHAR(64) NOT NULL,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    quantity INT NOT NULL,
    status VARCHAR(32) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_biz_id UNIQUE (biz_id)
);
CREATE INDEX IF NOT EXISTS idx_reservation_sku_code ON inventory_reservation_record (sku_code);
CREATE INDEX IF NOT EXISTS idx_reservation_warehouse_id ON inventory_reservation_record (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_reservation_status ON inventory_reservation_record (status);

CREATE TABLE IF NOT EXISTS inventory_operation_log (
    id BIGSERIAL PRIMARY KEY,
    operation_type VARCHAR(32) NOT NULL,
    sku_code VARCHAR(64) NOT NULL,
    warehouse_id BIGINT NOT NULL,
    quantity INT NOT NULL,
    biz_id VARCHAR(64),
    remark VARCHAR(512),
    operator_id VARCHAR(64),
    operator_name VARCHAR(128),
    quantity_before INT,
    quantity_after INT,
    operated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result VARCHAR(32) DEFAULT 'SUCCESS',
    error_message VARCHAR(1024),
    request_id VARCHAR(64),
    client_ip VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS idx_operation_log_sku_code ON inventory_operation_log (sku_code);
CREATE INDEX IF NOT EXISTS idx_operation_log_warehouse_id ON inventory_operation_log (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_operation_log_type ON inventory_operation_log (operation_type);
CREATE INDEX IF NOT EXISTS idx_operation_log_biz_id ON inventory_operation_log (biz_id);
CREATE INDEX IF NOT EXISTS idx_operation_log_operator_id ON inventory_operation_log (operator_id);
CREATE INDEX IF NOT EXISTS idx_operation_log_operated_at ON inventory_operation_log (operated_at);
CREATE INDEX IF NOT EXISTS idx_operation_log_result ON inventory_operation_log (result);

-- stock-check module

CREATE TABLE IF NOT EXISTS tb_stock_check_plan (
    id BIGSERIAL PRIMARY KEY,
    plan_no VARCHAR(64) NOT NULL,
    plan_name VARCHAR(128),
    check_type VARCHAR(32),
    warehouse_id BIGINT,
    warehouse_name VARCHAR(128),
    status VARCHAR(32),
    planned_start_time TIMESTAMP,
    planned_end_time TIMESTAMP,
    actual_start_time TIMESTAMP,
    actual_end_time TIMESTAMP,
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    remark VARCHAR(512),
    deleted SMALLINT DEFAULT 0,
    version INT DEFAULT 0,
    CONSTRAINT uk_scp_plan_no UNIQUE (plan_no)
);
CREATE INDEX IF NOT EXISTS idx_scp_warehouse_id ON tb_stock_check_plan (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_scp_status ON tb_stock_check_plan (status);

CREATE TABLE IF NOT EXISTS tb_stock_check_record (
    id BIGSERIAL PRIMARY KEY,
    plan_id BIGINT,
    plan_no VARCHAR(64),
    sku_code VARCHAR(64),
    product_name VARCHAR(128),
    location_code VARCHAR(32),
    warehouse_id BIGINT,
    book_quantity DECIMAL(19,2),
    check_quantity DECIMAL(19,2),
    difference_quantity DECIMAL(19,2),
    difference_type VARCHAR(32),
    status VARCHAR(32),
    checker VARCHAR(64),
    check_time TIMESTAMP,
    remark VARCHAR(512),
    source_system VARCHAR(32),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0,
    version INT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_scr_plan_id ON tb_stock_check_record (plan_id);
CREATE INDEX IF NOT EXISTS idx_scr_sku_code ON tb_stock_check_record (sku_code);
CREATE INDEX IF NOT EXISTS idx_scr_warehouse_id ON tb_stock_check_record (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_scr_status ON tb_stock_check_record (status);

CREATE TABLE IF NOT EXISTS tb_stock_check_diff (
    id BIGSERIAL PRIMARY KEY,
    record_id BIGINT,
    plan_id BIGINT,
    plan_no VARCHAR(64),
    sku_code VARCHAR(64),
    product_name VARCHAR(128),
    location_code VARCHAR(32),
    warehouse_id BIGINT,
    book_quantity DECIMAL(19,2),
    check_quantity DECIMAL(19,2),
    difference_quantity DECIMAL(19,2),
    difference_type VARCHAR(32),
    processing_status VARCHAR(32),
    approval_status VARCHAR(32),
    approver VARCHAR(64),
    approval_time TIMESTAMP,
    approval_remark VARCHAR(512),
    solution VARCHAR(256),
    processor VARCHAR(64),
    process_time TIMESTAMP,
    process_remark VARCHAR(512),
    source_system VARCHAR(32),
    creator VARCHAR(64),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64),
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted SMALLINT DEFAULT 0,
    version INT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_scd_record_id ON tb_stock_check_diff (record_id);
CREATE INDEX IF NOT EXISTS idx_scd_plan_id ON tb_stock_check_diff (plan_id);
CREATE INDEX IF NOT EXISTS idx_scd_sku_code ON tb_stock_check_diff (sku_code);
CREATE INDEX IF NOT EXISTS idx_scd_warehouse_id ON tb_stock_check_diff (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_scd_processing_status ON tb_stock_check_diff (processing_status);
CREATE INDEX IF NOT EXISTS idx_scd_approval_status ON tb_stock_check_diff (approval_status);
-- ---------- Part from warehouse-service ----------


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


-- ---------- Part from supplier-service ----------

-- schema.sql

-- =====================
-- V1: supplier init
-- =====================

CREATE TABLE IF NOT EXISTS supplier (
    id BIGSERIAL PRIMARY KEY,
    supplier_code VARCHAR(64) NOT NULL UNIQUE,
    supplier_name VARCHAR(128) NOT NULL,
    supplier_type VARCHAR(32), -- RAW_MATERIAL/PACKAGING/EQUIPMENT/SERVICES,
    business_license VARCHAR(64),
    tax_id VARCHAR(64),
    province VARCHAR(64),
    city VARCHAR(64),
    district VARCHAR(64),
    address VARCHAR(512),
    contact VARCHAR(128),
    phone VARCHAR(32),
    email VARCHAR(128),
    status VARCHAR(32) DEFAULT 'ACTIVE', -- ACTIVE/INACTIVE/SUSPENDED,
    credit_limit DECIMAL(15,2),
    payment_terms VARCHAR(64), -- NET_30/NET_60/NET_90,
    category VARCHAR(64),
    score INTEGER DEFAULT 0,
    level VARCHAR(32), -- A/B/C/D,
    assessment_level VARCHAR(32), -- EXCELLENT/GOOD/FAIR/POOR,
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
    shipment_method VARCHAR(32), -- EXPRESS/OCEAN/AIR/LAND,
    carrier_name VARCHAR(128),
    carrier_contact VARCHAR(64),
    tracking_number VARCHAR(128),
    vehicle_number VARCHAR(64),
    driver_name VARCHAR(64),
    driver_phone VARCHAR(32),
    total_quantity DECIMAL(15,3),
    total_weight DECIMAL(15,3),
    total_volume DECIMAL(15,3),
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/SUBMITTED/CONFIRMED/IN_TRANSIT/DELIVERED/CANCELLED,
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
    status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/SUBMITTED/CONFIRMED/REJECTED/PAID,
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
    status VARCHAR(32), -- PENDING/CONFIRMED/REJECTED,
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
    supplier_code VARCHAR(50),
    supplier_name VARCHAR(200),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
 on_time_delivery_rate DECIMAL(5, 2), -- (0-100),
 quality_pass_rate DECIMAL(5, 2), -- (0-100),
 price_competitiveness_score DECIMAL(5, 2), -- (0-100),
 overall_score DECIMAL(5, 2), -- (0-100),
 assessment_level VARCHAR(10), -- (A/B/C/D),
    total_orders INT,
    on_time_delivery_count INT,
    qualified_count INT,
    total_quality_check_count INT,
    market_avg_price DECIMAL(18, 2),
    supplier_price DECIMAL(18, 2),
    remark VARCHAR(500),
    assessor VARCHAR(100),
    assessment_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_supplier_performance_supplier_id ON supplier_performance (supplier_id);
CREATE INDEX IF NOT EXISTS idx_supplier_performance_period ON supplier_performance (period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_supplier_performance_assessment_level ON supplier_performance (assessment_level);

-- ---------- Part from dom-service ----------

-- dom-service

CREATE TABLE IF NOT EXISTS dom_order (
    id BIGSERIAL PRIMARY KEY,
    dom_order_no VARCHAR(64) NOT NULL,
    original_order_no VARCHAR(64),
    customer_id VARCHAR(64),
    customer_name VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    total_amount DECIMAL(18,2),
    currency VARCHAR(8) DEFAULT 'CNY',
    priority INT DEFAULT 0,
    sourcing_strategy VARCHAR(32),
    region VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_dom_order_no ON dom_order (dom_order_no);
CREATE INDEX IF NOT EXISTS idx_original_order_no ON dom_order (original_order_no);
CREATE INDEX IF NOT EXISTS idx_status ON dom_order (status);

CREATE TABLE IF NOT EXISTS dom_order_line (
    id BIGSERIAL PRIMARY KEY,
    dom_order_id BIGINT NOT NULL,
    sku_code VARCHAR(64) NOT NULL, -- SKU
    sku_name VARCHAR(256),
    quantity INT NOT NULL DEFAULT 0,
    fulfilled_quantity INT DEFAULT 0,
    warehouse_id BIGINT, -- ID
    warehouse_name VARCHAR(128),
    unit_price DECIMAL(18,2),
    status VARCHAR(32) DEFAULT 'PENDING',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dom_order_id ON dom_order_line (dom_order_id);
CREATE INDEX IF NOT EXISTS idx_sku_code ON dom_order_line (sku_code);

CREATE TABLE IF NOT EXISTS fulfillment_plan (
    id BIGSERIAL PRIMARY KEY,
    dom_order_id BIGINT NOT NULL,
    dom_order_no VARCHAR(64),
    total_lines INT DEFAULT 0,
    fulfilled_lines INT DEFAULT 0,
    partially_fulfilled_lines INT DEFAULT 0,
    unfulfilled_lines INT DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'DRAFT',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dom_order_id ON fulfillment_plan (dom_order_id);
CREATE INDEX IF NOT EXISTS idx_dom_order_no ON fulfillment_plan (dom_order_no);
CREATE INDEX IF NOT EXISTS idx_status ON fulfillment_plan (status);

CREATE TABLE IF NOT EXISTS sourcing_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_name VARCHAR(128) NOT NULL,
    rule_type VARCHAR(32) NOT NULL,
    priority INT DEFAULT 0,
    warehouse_ids VARCHAR(512),
    region VARCHAR(64),
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rule_type ON sourcing_rule (rule_type);
CREATE INDEX IF NOT EXISTS idx_region ON sourcing_rule (region);
CREATE INDEX IF NOT EXISTS idx_enabled ON sourcing_rule (enabled);

