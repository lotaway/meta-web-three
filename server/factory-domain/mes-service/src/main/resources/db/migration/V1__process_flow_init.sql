-- 流程引擎表结构
-- Flowable流程设计器基础表

-- 流程模板表
CREATE TABLE IF NOT EXISTS mes_process_flow_template (
    id BIGINT PRIMARY KEY,
    template_code VARCHAR(64) NOT NULL UNIQUE,
    template_name VARCHAR(128) NOT NULL,
    description VARCHAR(512),
    version INT DEFAULT 1,
    flow_data TEXT, -- JSON存储流程图数据
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PUBLISHED/ARCHIVED
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

-- 流程节点类型表
CREATE TABLE IF NOT EXISTS mes_process_node_type (
    id BIGINT PRIMARY KEY,
    node_type_code VARCHAR(64) NOT NULL UNIQUE,
    node_type_name VARCHAR(128) NOT NULL,
    category VARCHAR(32) NOT NULL, -- START/END/TASK/GATEWAY/SERVICE/EQUIPMENT/SUB_PROCESS
    icon VARCHAR(32),
    config_schema TEXT, -- JSON Schema for node configuration
    description VARCHAR(512),
    enabled BOOLEAN DEFAULT TRUE,
    sort_order INT DEFAULT 0,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

-- 流程实例表
CREATE TABLE IF NOT EXISTS mes_process_flow_instance (
    id BIGINT PRIMARY KEY,
    instance_code VARCHAR(64) NOT NULL UNIQUE,
    template_id BIGINT NOT NULL,
    template_name VARCHAR(128),
    business_type VARCHAR(64), -- WORK_ORDER/QC/MAINTENANCE/ANDON etc.
    business_key VARCHAR(128), -- Reference to business entity
    current_node_id VARCHAR(64),
    current_node_name VARCHAR(128),
    status VARCHAR(32) DEFAULT 'RUNNING', -- RUNNING/SUSPENDED/COMPLETED/TERMINATED
    flow_data TEXT,
    started_at TIMESTAMP,
    started_by BIGINT,
    completed_at TIMESTAMP,
    completed_by BIGINT,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (template_id) REFERENCES mes_process_flow_template(id)
);

-- 创建索引
CREATE INDEX idx_template_status ON mes_process_flow_template(status);
CREATE INDEX idx_template_code ON mes_process_flow_template(template_code);
CREATE INDEX idx_instance_code ON mes_process_flow_instance(instance_code);
CREATE INDEX idx_instance_business ON mes_process_flow_instance(business_type, business_key);
CREATE INDEX idx_instance_status ON mes_process_flow_instance(status);

-- 初始化预置节点类型
INSERT INTO mes_process_node_type (id, node_type_code, node_type_name, category, icon, sort_order, description, enabled, deleted)
VALUES 
    (1, 'start', '开始', 'START', '▶', 1, '流程开始节点', TRUE, FALSE),
    (2, 'end', '结束', 'END', '■', 2, '流程结束节点', TRUE, FALSE),
    (3, 'manual_task', '人工任务', 'TASK', '👤', 3, '需要人工处理的任务节点', TRUE, FALSE),
    (4, 'equipment_task', '设备交互', 'EQUIPMENT', '⚙', 4, '与设备进行数据交互的节点', TRUE, FALSE),
    (5, 'data_process', '数据处理', 'SERVICE', '⚡', 5, '数据处理和服务调用节点', TRUE, FALSE),
    (6, 'system_integration', '系统集成', 'SERVICE', '🔗', 6, '与外部系统集成的节点', TRUE, FALSE),
    (7, 'exclusive_gateway', '排他网关', 'GATEWAY', '◇', 7, '根据条件选择分支', TRUE, FALSE),
    (8, 'parallel_gateway', '并行网关', 'GATEWAY', '◆', 8, '并行执行多个分支', TRUE, FALSE),
    (9, 'inclusive_gateway', '包容网关', 'GATEWAY', '◈', 9, '根据条件并行执行分支', TRUE, FALSE),
    (10, 'sub_process', '子流程', 'SUB_PROCESS', '🔄', 10, '调用子流程', TRUE, FALSE)
ON CONFLICT (node_type_code) DO NOTHING;