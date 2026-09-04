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