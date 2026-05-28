-- V8: Non-Conformance Disposition (不合格处置流程)
CREATE TABLE IF NOT EXISTS mes_qc_non_conformance (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    disposition_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'Disposition Code',
    disposition_name VARCHAR(100) NOT NULL COMMENT 'Disposition Name',
    type VARCHAR(20) NOT NULL COMMENT 'Type: SCRAP, REWORK, REPAIR, RETURN, USE_AS_IS, SPECIAL_ACCEPTANCE',
    steps_json TEXT COMMENT 'Steps JSON',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT 'Is Enabled',
    sort_order INT DEFAULT 0 COMMENT 'Sort Order',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated At',
    INDEX idx_disposition_code (disposition_code),
    INDEX idx_type (type),
    INDEX idx_is_enabled (is_enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='QC Non-Conformance Disposition';