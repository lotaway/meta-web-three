-- Digital Twin tables that were lost when the standalone digital-twin-service
-- was consolidated into factory-service. The runtime reported:
--   ERROR: relation "devices" does not exist
--   ERROR: relation "production_lines" does not exist
-- This script is idempotent (IF NOT EXISTS) so it is safe to run on an existing
-- development database without re-running the full schema.sql (whose ALTER TABLE
-- constraint statements are not idempotent).

-- Workshops (车间)
CREATE TABLE IF NOT EXISTS workshops (
    id BIGSERIAL PRIMARY KEY,
    workshop_code VARCHAR(50) NOT NULL UNIQUE,
    workshop_name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'PLANNING',
    area INTEGER,
    location VARCHAR(200),
    center_x DOUBLE PRECISION,
    center_y DOUBLE PRECISION,
    width DOUBLE PRECISION,
    length DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Production Lines (产线)
CREATE TABLE IF NOT EXISTS production_lines (
    id BIGSERIAL PRIMARY KEY,
    line_code VARCHAR(50) NOT NULL UNIQUE,
    line_name VARCHAR(100) NOT NULL,
    workshop_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'IDLE',
    capacity INTEGER NOT NULL DEFAULT 0,
    current_output INTEGER NOT NULL DEFAULT 0,
    efficiency DOUBLE PRECISION NOT NULL DEFAULT 0,
    product_types VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Devices (设备)
CREATE TABLE IF NOT EXISTS devices (
    id BIGSERIAL PRIMARY KEY,
    device_code VARCHAR(50) NOT NULL UNIQUE,
    device_name VARCHAR(100) NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    workshop_id VARCHAR(50),
    production_line_id VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'OFFLINE',
    position_x DOUBLE PRECISION,
    position_y DOUBLE PRECISION DEFAULT 0.25,
    position_z DOUBLE PRECISION,
    rotation_y DOUBLE PRECISION DEFAULT 0,
    ip_address VARCHAR(45),
    mac_address VARCHAR(17),
    mqtt_topic VARCHAR(200),
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts (告警)
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_code VARCHAR(50) NOT NULL UNIQUE,
    device_code VARCHAR(50) NOT NULL,
    workshop_id VARCHAR(50),
    level VARCHAR(20) NOT NULL,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description VARCHAR(1000),
    status VARCHAR(20) NOT NULL DEFAULT 'TRIGGERED',
    solution VARCHAR(1000),
    acknowledged_by VARCHAR(50),
    resolved_by VARCHAR(50),
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alert Rules (告警规则)
CREATE TABLE IF NOT EXISTS alert_rules (
    id BIGSERIAL PRIMARY KEY,
    rule_code VARCHAR(50) NOT NULL UNIQUE,
    rule_name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    device_type VARCHAR(50),
    device_code VARCHAR(50),
    workshop_id VARCHAR(50),
    metric_type VARCHAR(50),
    operator VARCHAR(20),
    threshold_value DOUBLE PRECISION,
    duration_seconds INTEGER,
    level VARCHAR(20),
    alert_type VARCHAR(50),
    title_template VARCHAR(200),
    description_template VARCHAR(500),
    enabled BOOLEAN DEFAULT TRUE,
    cooldown_seconds INTEGER,
    max_alerts_per_hour INTEGER,
    notification_channels VARCHAR(500),
    created_by VARCHAR(50),
    updated_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Digital twin indexes
CREATE INDEX IF NOT EXISTS idx_dt_workshops_status ON workshops(status);
CREATE INDEX IF NOT EXISTS idx_dt_production_lines_workshop ON production_lines(workshop_id);
CREATE INDEX IF NOT EXISTS idx_dt_production_lines_status ON production_lines(status);
CREATE INDEX IF NOT EXISTS idx_dt_devices_workshop ON devices(workshop_id);
CREATE INDEX IF NOT EXISTS idx_dt_devices_line ON devices(production_line_id);
CREATE INDEX IF NOT EXISTS idx_dt_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_dt_alerts_device ON alerts(device_code);
CREATE INDEX IF NOT EXISTS idx_dt_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_dt_alerts_level ON alerts(level);
CREATE INDEX IF NOT EXISTS idx_dt_alerts_occurred ON alerts(occurred_at);
CREATE INDEX IF NOT EXISTS idx_dt_alert_rules_enabled ON alert_rules(enabled);
