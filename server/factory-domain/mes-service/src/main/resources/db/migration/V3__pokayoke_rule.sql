-- 防错规则引擎表
CREATE TABLE IF NOT EXISTS mes_pokayoke_rule (
    id BIGINT PRIMARY KEY,
    rule_code VARCHAR(64) NOT NULL UNIQUE,
    rule_name VARCHAR(128) NOT NULL,
    rule_type VARCHAR(32) NOT NULL, -- MATERIAL_CHECK/SEQUENCE_CHECK/PARAMETER_CHECK/QUALITY_CHECK
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/ACTIVE/INACTIVE
    workstation_id VARCHAR(64),
    process_code VARCHAR(64),
    product_code VARCHAR(64),
    trigger_condition VARCHAR(64), -- ON_MATERIAL_SCAN/ON_TASK_START/ON_TASK_COMPLETE/ON_PARAMETER_RECORD/MANUAL_TRIGGER
    actions_json TEXT, -- JSON array of CheckAction
    priority INT DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    description VARCHAR(512),
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_pokayoke_status ON mes_pokayoke_rule(status);
CREATE INDEX idx_pokayoke_workstation ON mes_pokayoke_rule(workstation_id);
CREATE INDEX idx_pokayoke_process ON mes_pokayoke_rule(process_code);