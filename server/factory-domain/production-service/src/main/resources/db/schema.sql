-- 工位绑定表（设备、工具、人员）
CREATE TABLE IF NOT EXISTS work_station_binding (
    id BIGSERIAL PRIMARY KEY,
    workstation_code VARCHAR(50) NOT NULL COMMENT '工位编码',
    binding_type VARCHAR(20) NOT NULL COMMENT '绑定类型: EQUIPMENT, TOOL, PERSONNEL',
    target_code VARCHAR(50) NOT NULL COMMENT '目标编码(设备/工具/人员编码)',
    target_name VARCHAR(100) COMMENT '目标名称',
    target_type VARCHAR(50) COMMENT '目标类型',
    quantity INTEGER DEFAULT 1 COMMENT '数量',
    is_primary BOOLEAN DEFAULT FALSE COMMENT '是否主绑定',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    remark VARCHAR(500) COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workstation_code, binding_type, target_code)
);

CREATE INDEX idx_ws_binding_workstation ON work_station_binding(workstation_code);
CREATE INDEX idx_ws_binding_type ON work_station_binding(binding_type);
CREATE INDEX idx_ws_binding_target ON work_station_binding(target_code);
CREATE INDEX idx_ws_binding_status ON work_station_binding(status);