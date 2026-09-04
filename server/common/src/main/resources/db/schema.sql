-- Operation Log Table for Audit Trail
-- Records who operated on what data at what time

CREATE TABLE IF NOT EXISTS tb_operation_log (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    username VARCHAR(64),
    operation VARCHAR(128) NOT NULL,
    method VARCHAR(256),
    params TEXT,
    ip VARCHAR(64),
    operation_time TIMESTAMP NOT NULL,
    execution_time BIGINT,
    status VARCHAR(32),
    error_message TEXT,
    entity_type VARCHAR(64),
    entity_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_operation_log_user_id ON tb_operation_log (user_id);
CREATE INDEX IF NOT EXISTS idx_operation_log_operation ON tb_operation_log (operation);
CREATE INDEX IF NOT EXISTS idx_operation_log_status ON tb_operation_log (status);
CREATE INDEX IF NOT EXISTS idx_operation_log_operation_time ON tb_operation_log (operation_time);
CREATE INDEX IF NOT EXISTS idx_operation_log_entity ON tb_operation_log (entity_type, entity_id);

-- Audit Log Table for general auditing
CREATE TABLE IF NOT EXISTS tb_audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT,
    username VARCHAR(64),
    operation_type VARCHAR(64),
    resource_type VARCHAR(64),
    resource_id VARCHAR(64),
    description TEXT,
    result VARCHAR(32),
    error_message TEXT,
    ip_address VARCHAR(64),
    request_url VARCHAR(512),
    request_method VARCHAR(16),
    duration BIGINT,
    operation_time TIMESTAMP NOT NULL,
    request_params TEXT,
    response_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON tb_audit_log (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation_type ON tb_audit_log (operation_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON tb_audit_log (resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation_time ON tb_audit_log (operation_time);