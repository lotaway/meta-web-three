-- inventory-service

CREATE TABLE IF NOT EXISTS inventory (
    id BIGSERIAL PRIMARY KEY,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
    total_quantity INT DEFAULT 0
    available_quantity INT DEFAULT 0
    reserved_quantity INT DEFAULT 0
    defective_quantity INT DEFAULT 0
    unit_cost DECIMAL(10,2) DEFAULT 0
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    version INT DEFAULT 0
    CONSTRAINT uk_sku_warehouse UNIQUE (sku_code, warehouse_id)
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON inventory (warehouse_id);

CREATE TABLE IF NOT EXISTS inventory_record (
    id BIGSERIAL PRIMARY KEY,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
 biz_type VARCHAR(32) NOT NULL, -- RESERVE/CONFIRM/CANCEL/INCREASE/DECREASE,
 biz_id VARCHAR(64), -- ID,
    quantity INT NOT NULL
    before_quantity INT
    after_quantity INT
    remark VARCHAR(512)
    operator VARCHAR(64)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory_record (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON inventory_record (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_biz_type ON inventory_record (biz_type);
CREATE INDEX IF NOT EXISTS idx_biz_id ON inventory_record (biz_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON inventory_record (created_at);

CREATE TABLE IF NOT EXISTS replenishment_recommendation (
    id BIGSERIAL PRIMARY KEY,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
    current_stock INT
    safety_stock INT
    lead_time_days INT
    average_daily_sales INT
    recommended_quantity INT
 recommendation_type VARCHAR(32) DEFAULT 'AUTO', -- AUTO/MANUAL,
 status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/APPROVED/REJECTED,
    generated_at TIMESTAMP
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON replenishment_recommendation (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON replenishment_recommendation (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_status ON replenishment_recommendation (status);
CREATE INDEX IF NOT EXISTS idx_generated_at ON replenishment_recommendation (generated_at);

CREATE TABLE IF NOT EXISTS demand_forecast (
    id BIGSERIAL PRIMARY KEY,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
    forecast_period_days INT
    predicted_quantity INT
 confidence_level INT, -- (0-100),
 forecast_method VARCHAR(32) DEFAULT 'SMA', -- SMA/WMA/EXPONENTIAL_SMOOTHING,
    forecast_start_date DATE
    forecast_end_date DATE
 status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/APPROVED/REJECTED,
    generated_at TIMESTAMP
    notes VARCHAR(512)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON demand_forecast (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON demand_forecast (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_status ON demand_forecast (status);
CREATE INDEX IF NOT EXISTS idx_forecast_dates ON demand_forecast (forecast_start_date, forecast_end_date);

CREATE TABLE IF NOT EXISTS inventory_batch (
    id BIGSERIAL PRIMARY KEY,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
    batch_no VARCHAR(64) NOT NULL
    quantity INT NOT NULL
    available_quantity INT DEFAULT 0
    reserved_quantity INT DEFAULT 0
    picked_quantity INT DEFAULT 0
    inbound_date TIMESTAMP
    production_date DATE
    expiry_date DATE
    unit_cost DECIMAL(10,2) DEFAULT 0
    location_code VARCHAR(32)
 status VARCHAR(32) DEFAULT 'AVAILABLE', -- AVAILABLE/EXHAUSTED/LOCKED/EXPIRED,
    remark VARCHAR(512)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    version INT DEFAULT 0
    CONSTRAINT uk_batch UNIQUE (sku_code, warehouse_id, batch_no)
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory_batch (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON inventory_batch (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_batch_no ON inventory_batch (batch_no);
CREATE INDEX IF NOT EXISTS idx_inbound_date ON inventory_batch (inbound_date);
CREATE INDEX IF NOT EXISTS idx_expiry_date ON inventory_batch (expiry_date);
CREATE INDEX IF NOT EXISTS idx_status ON inventory_batch (status);

CREATE TABLE IF NOT EXISTS outbound_strategy (
    id BIGSERIAL PRIMARY KEY,
    strategy_code VARCHAR(64) NOT NULL
    strategy_name VARCHAR(128) NOT NULL
 strategy_type VARCHAR(32) NOT NULL, -- FIFO/LIFO/SPECIFIC_BATCH,
 warehouse_id BIGINT, -- ID,
    warehouse_code VARCHAR(32)
 sku_code VARCHAR(64), -- SKU,
 sku_code_pattern VARCHAR(128), -- SKU,
    priority INT DEFAULT 100
 specific_batch_no VARCHAR(64), -- SPECIFIC_BATCH,
    is_active BOOLEAN DEFAULT TRUE
    remark VARCHAR(512)
    creator VARCHAR(64)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    version INT DEFAULT 0
    CONSTRAINT uk_strategy_code UNIQUE (strategy_code)
)
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON outbound_strategy (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_sku_code ON outbound_strategy (sku_code);
CREATE INDEX IF NOT EXISTS idx_strategy_type ON outbound_strategy (strategy_type);
CREATE INDEX IF NOT EXISTS idx_is_active ON outbound_strategy (is_active);
CREATE INDEX IF NOT EXISTS idx_priority ON outbound_strategy (priority);

CREATE TABLE IF NOT EXISTS inventory_alert_config (
    id BIGSERIAL PRIMARY KEY,
    config_code VARCHAR(64) NOT NULL
    warehouse_code VARCHAR(32)
 sku_code VARCHAR(64), -- SKU,
    safety_stock_threshold INT
 level VARCHAR(32) DEFAULT 'WARNING', -- INFO/WARNING/ERROR/CRITICAL,
    enabled BOOLEAN DEFAULT TRUE
    cooldown_minutes INT DEFAULT 60
 notification_channels VARCHAR(128), -- EMAIL,SMS,IN_APP,DINGTALK,
    notify_users VARCHAR(512)
    created_by VARCHAR(64)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_by VARCHAR(64)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    version INT DEFAULT 0
    CONSTRAINT uk_config_code UNIQUE (config_code)
)
CREATE INDEX IF NOT EXISTS idx_warehouse_code ON inventory_alert_config (warehouse_code);
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory_alert_config (sku_code);
CREATE INDEX IF NOT EXISTS idx_enabled ON inventory_alert_config (enabled);

CREATE TABLE IF NOT EXISTS inventory_alert (
    id BIGSERIAL PRIMARY KEY,
    alert_code VARCHAR(64) NOT NULL
    warehouse_code VARCHAR(32) NOT NULL
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 alert_type VARCHAR(32) NOT NULL, -- LOW_STOCK/OVERSTOCK/EXPIRING_SOON/EXPIRED,
 level VARCHAR(32) DEFAULT 'WARNING', -- INFO/WARNING/ERROR/CRITICAL,
    title VARCHAR(256) NOT NULL
    description VARCHAR(1024)
    current_quantity INT
    threshold_value INT
 status VARCHAR(32) DEFAULT 'TRIGGERED', -- TRIGGERED/ACKNOWLEDGED/IN_PROGRESS/RESOLVED/CLOSED,
    solution VARCHAR(1024)
    acknowledged_by VARCHAR(64)
    acknowledged_at TIMESTAMP
    resolved_by VARCHAR(64)
    resolved_at TIMESTAMP
    occurred_at TIMESTAMP NOT NULL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    version INT DEFAULT 0
    CONSTRAINT uk_alert_code UNIQUE (alert_code)
)
CREATE INDEX IF NOT EXISTS idx_warehouse_code ON inventory_alert (warehouse_code);
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory_alert (sku_code);
CREATE INDEX IF NOT EXISTS idx_alert_type ON inventory_alert (alert_type);
CREATE INDEX IF NOT EXISTS idx_level ON inventory_alert (level);
CREATE INDEX IF NOT EXISTS idx_status ON inventory_alert (status);
CREATE INDEX IF NOT EXISTS idx_occurred_at ON inventory_alert (occurred_at);
CREATE INDEX IF NOT EXISTS idx_created_at ON inventory_alert (created_at);

CREATE TABLE IF NOT EXISTS inventory_reservation_record (
    id BIGSERIAL PRIMARY KEY,
 biz_id VARCHAR(64) NOT NULL, -- ID,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
    quantity INT NOT NULL
 status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/CONFIRMED/CANCELLED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    CONSTRAINT uk_biz_id UNIQUE (biz_id)
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory_reservation_record (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON inventory_reservation_record (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_status ON inventory_reservation_record (status);

CREATE TABLE IF NOT EXISTS inventory_operation_log (
    id BIGSERIAL PRIMARY KEY,
 operation_type VARCHAR(32) NOT NULL, -- RESERVE/CONFIRM/CANCEL/INCREASE/DECREASE,
 sku_code VARCHAR(64) NOT NULL, -- SKU,
 warehouse_id BIGINT NOT NULL, -- ID,
    quantity INT NOT NULL
 biz_id VARCHAR(64), -- ID,
    remark VARCHAR(512)
 operator_id VARCHAR(64), -- ID,
    operator_name VARCHAR(128)
    quantity_before INT
    quantity_after INT
    operated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
 result VARCHAR(32) DEFAULT 'SUCCESS', -- SUCCESS/FAILED,
    error_message VARCHAR(1024)
 request_id VARCHAR(64), -- ID,
 client_ip VARCHAR(64), -- IP
)
CREATE INDEX IF NOT EXISTS idx_sku_code ON inventory_operation_log (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON inventory_operation_log (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_operation_type ON inventory_operation_log (operation_type);
CREATE INDEX IF NOT EXISTS idx_biz_id ON inventory_operation_log (biz_id);
CREATE INDEX IF NOT EXISTS idx_operator_id ON inventory_operation_log (operator_id);
CREATE INDEX IF NOT EXISTS idx_operated_at ON inventory_operation_log (operated_at);
CREATE INDEX IF NOT EXISTS idx_result ON inventory_operation_log (result);
