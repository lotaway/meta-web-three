-- V9: SPC Control Chart (SPC控制图)
CREATE TABLE IF NOT EXISTS mes_qc_spc_control_chart (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chart_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'Chart Code',
    chart_name VARCHAR(100) NOT NULL COMMENT 'Chart Name',
    chart_type VARCHAR(20) NOT NULL COMMENT 'Chart Type: XBAR_R, XBAR_S, X_MR, P_CHART, NP_CHART, C_CHART, CU_CHART',
    parameter_code VARCHAR(50) COMMENT 'Parameter Code',
    limits_json TEXT COMMENT 'Control Limits JSON',
    alarm_rules_json TEXT COMMENT 'Alarm Rules JSON',
    sampling_config_json TEXT COMMENT 'Sampling Config JSON',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT 'Is Enabled',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated At',
    INDEX idx_chart_code (chart_code),
    INDEX idx_chart_type (chart_type),
    INDEX idx_parameter_code (parameter_code),
    INDEX idx_is_enabled (is_enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='QC SPC Control Chart';