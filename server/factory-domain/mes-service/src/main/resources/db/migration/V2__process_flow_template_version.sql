-- 流程模板版本管理表
CREATE TABLE IF NOT EXISTS mes_process_flow_template_version (
    id BIGINT PRIMARY KEY,
    template_id BIGINT NOT NULL,
    version INT NOT NULL,
    template_code VARCHAR(64) NOT NULL,
    template_name VARCHAR(128) NOT NULL,
    description VARCHAR(512),
    flow_data TEXT,
    status VARCHAR(32),
    change_description VARCHAR(512),
    is_current_version BOOLEAN DEFAULT FALSE,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (template_id) REFERENCES mes_process_flow_template(id)
);

CREATE INDEX idx_version_template ON mes_process_flow_template_version(template_id);
CREATE INDEX idx_version_template_version ON mes_process_flow_template_version(template_id, version);