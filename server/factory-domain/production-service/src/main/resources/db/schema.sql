CREATE TABLE IF NOT EXISTS work_station_binding (
    id BIGSERIAL PRIMARY KEY,
    workstation_code VARCHAR(50) NOT NULL,
 binding_type VARCHAR(20) NOT NULL, -- : EQUIPMENT, TOOL, PERSONNEL,
 target_code VARCHAR(50) NOT NULL, -- (//),
    target_name VARCHAR(100),
    target_type VARCHAR(50),
    quantity INTEGER DEFAULT 1,
    is_primary BOOLEAN DEFAULT FALSE,
 status VARCHAR(20) DEFAULT 'ACTIVE', -- : ACTIVE, INACTIVE,
    remark VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workstation_code, binding_type, target_code)
);

CREATE INDEX idx_ws_binding_workstation ON work_station_binding(workstation_code);
CREATE INDEX idx_ws_binding_type ON work_station_binding(binding_type);
CREATE INDEX idx_ws_binding_target ON work_station_binding(target_code);
CREATE INDEX idx_ws_binding_status ON work_station_binding(status);