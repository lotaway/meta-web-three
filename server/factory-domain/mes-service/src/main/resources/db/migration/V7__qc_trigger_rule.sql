-- V7: Qc Trigger Rule (检验触发规则)
CREATE TABLE IF NOT EXISTS mes_qc_trigger_rule (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    rule_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'Rule Code',
    rule_name VARCHAR(100) NOT NULL COMMENT 'Rule Name',
    trigger_type VARCHAR(20) NOT NULL COMMENT 'Trigger Type: BY_BATCH, BY_TIME, BY_QUANTITY, BY_EVENT, MANUAL',
    target_object VARCHAR(100) COMMENT 'Target Object: WORK_ORDER, PRODUCTION_TASK, PROCESS_STEP',
    condition_json TEXT COMMENT 'Trigger Condition JSON',
    inspection_type VARCHAR(50) COMMENT 'Inspection Type',
    inspection_plan_code VARCHAR(50) COMMENT 'Inspection Plan Code',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT 'Is Enabled',
    priority INT DEFAULT 0 COMMENT 'Priority',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated At',
    INDEX idx_rule_code (rule_code),
    INDEX idx_trigger_type (trigger_type),
    INDEX idx_is_enabled (is_enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='QC Trigger Rule';