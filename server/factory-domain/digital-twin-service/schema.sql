-- Digital Twin Service - Database Schema
-- Compatible with H2, PostgreSQL, and MySQL.

-- Workshops (车间)
CREATE TABLE IF NOT EXISTS workshops (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    workshop_code VARCHAR(50) NOT NULL UNIQUE,
    workshop_name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'PLANNING',
    area DECIMAL(10, 2),
    location VARCHAR(200),
    center_x DECIMAL(10, 2),
    center_y DECIMAL(10, 2),
    width DECIMAL(10, 2),
    length DECIMAL(10, 2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_workshop_status CHECK (status IN ('PLANNING', 'CONSTRUCTION', 'OPERATING', 'MAINTENANCE', 'DECOMMISSIONED'))
);

-- Production Lines (产线)
CREATE TABLE IF NOT EXISTS production_lines (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    line_code VARCHAR(50) NOT NULL UNIQUE,
    line_name VARCHAR(100) NOT NULL,
    workshop_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'IDLE',
    capacity INT NOT NULL DEFAULT 0,
    current_output INT NOT NULL DEFAULT 0,
    efficiency DECIMAL(5, 2) NOT NULL DEFAULT 0.00,
    product_types VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_line_status CHECK (status IN ('IDLE', 'RUNNING', 'PAUSED', 'MAINTENANCE', 'BROKEN_DOWN'))
);

-- Devices (设备)
CREATE TABLE IF NOT EXISTS devices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    device_code VARCHAR(50) NOT NULL UNIQUE,
    device_name VARCHAR(100) NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    workshop_id VARCHAR(50),
    production_line_id VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'OFFLINE',
    position_x DECIMAL(10, 2),
    position_y DECIMAL(10, 2) DEFAULT 0.25,
    position_z DECIMAL(10, 2),
    rotation_y DECIMAL(10, 5) DEFAULT 0,
    ip_address VARCHAR(45),
    mac_address VARCHAR(17),
    mqtt_topic VARCHAR(200),
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_device_status CHECK (status IN ('ONLINE', 'OFFLINE', 'RUNNING', 'IDLE', 'WARNING', 'ERROR', 'MAINTENANCE'))
);

-- Alerts (告警)
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
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
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_alert_level CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    CONSTRAINT chk_alert_status CHECK (status IN ('TRIGGERED', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED'))
);

-- Alert Rules (告警规则)
CREATE TABLE IF NOT EXISTS alert_rules (
    id BIGINT PRIMARY KEY,
    rule_code VARCHAR(50) NOT NULL UNIQUE,
    rule_name VARCHAR(200) NOT NULL,
    description VARCHAR(1000),
    device_type VARCHAR(50),
    device_code VARCHAR(50),
    workshop_id VARCHAR(50),
    metric_type VARCHAR(30),
    operator VARCHAR(20),
    threshold_value DECIMAL(15, 4),
    duration_seconds INT,
    level VARCHAR(20) NOT NULL DEFAULT 'WARNING',
    alert_type VARCHAR(30),
    title_template VARCHAR(500),
    description_template VARCHAR(1000),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    cooldown_seconds INT NOT NULL DEFAULT 300,
    max_alerts_per_hour INT NOT NULL DEFAULT 10,
    notification_channels VARCHAR(500),
    created_by VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_alert_rule_level CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

-- Production Log (产量日志 - 时序数据)
CREATE TABLE IF NOT EXISTS production_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    line_code VARCHAR(50) NOT NULL,
    output INT NOT NULL DEFAULT 0,
    efficiency DECIMAL(5, 2) NOT NULL DEFAULT 0.00,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Telemetry Log (遥测日志 - 时序数据)
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    device_code VARCHAR(50) NOT NULL,
    temperature DECIMAL(8, 2),
    humidity DECIMAL(5, 2),
    pressure DECIMAL(8, 2),
    vibration DECIMAL(8, 4),
    power DECIMAL(10, 2),
    rpm INT,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_devices_workshop ON devices(workshop_id);
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_alerts_device ON alerts(device_code);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
CREATE INDEX IF NOT EXISTS idx_alerts_occurred ON alerts(occurred_at);
CREATE INDEX IF NOT EXISTS idx_production_logs_line ON production_logs(line_code);
CREATE INDEX IF NOT EXISTS idx_production_logs_time ON production_logs(recorded_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_logs_device ON telemetry_logs(device_code);
CREATE INDEX IF NOT EXISTS idx_telemetry_logs_time ON telemetry_logs(recorded_at);
