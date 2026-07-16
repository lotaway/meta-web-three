-- Operation Log Table for Audit Trail
-- Records who operated on what data at what time

CREATE TABLE IF NOT EXISTS tb_operation_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL COMMENT 'User ID who performed the operation',
    username VARCHAR(64) COMMENT 'Username',
    operation VARCHAR(128) NOT NULL COMMENT 'Operation description',
    method VARCHAR(256) COMMENT 'Full method signature',
    params TEXT COMMENT 'Method parameters (JSON)',
    ip VARCHAR(64) COMMENT 'Client IP address',
    operation_time DATETIME NOT NULL COMMENT 'Operation timestamp',
    execution_time BIGINT COMMENT 'Execution time in milliseconds',
    status VARCHAR(32) COMMENT 'Operation status (SUCCESS/FAILURE/ERROR)',
    error_message TEXT COMMENT 'Error message if operation failed',
    entity_type VARCHAR(64) COMMENT 'Entity type being operated on',
    entity_id BIGINT COMMENT 'Entity ID being operated on',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_user_id (user_id),
    INDEX idx_operation (operation),
    INDEX idx_status (status),
    INDEX idx_operation_time (operation_time),
    INDEX idx_entity (entity_type, entity_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Operation audit log table';

-- Audit Log Table for general auditing
CREATE TABLE IF NOT EXISTS tb_audit_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
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
    operation_time DATETIME NOT NULL,
    request_params TEXT,
    response_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_audit_user_id (user_id),
    INDEX idx_audit_operation_type (operation_type),
    INDEX idx_audit_resource (resource_type, resource_id),
    INDEX idx_audit_operation_time (operation_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Audit log table';
