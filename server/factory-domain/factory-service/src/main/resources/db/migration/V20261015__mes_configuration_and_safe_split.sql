-- Additive MES persistence. Apply manually with psql -v ON_ERROR_STOP=1.
-- Existing work-order/equipment tables are not replaced.
BEGIN;

CREATE TABLE IF NOT EXISTS mes_status_machine (
    id BIGSERIAL PRIMARY KEY,
    machine_code TEXT NOT NULL UNIQUE,
    machine_name TEXT,
    entity_type TEXT NOT NULL,
    description TEXT,
    initial_status TEXT,
    is_default BOOLEAN DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS mes_status_config (
    id BIGSERIAL PRIMARY KEY,
    machine_id BIGINT NOT NULL REFERENCES mes_status_machine(id),
    status_code TEXT NOT NULL,
    status_name TEXT,
    status_category TEXT,
    is_initial BOOLEAN DEFAULT FALSE,
    is_final BOOLEAN DEFAULT FALSE,
    color TEXT,
    icon TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(machine_id, status_code)
);
CREATE TABLE IF NOT EXISTS mes_status_transition (
    id BIGSERIAL PRIMARY KEY,
    machine_id BIGINT NOT NULL REFERENCES mes_status_machine(id),
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    transition_action TEXT,
    condition_expression TEXT,
    event_code TEXT,
    is_auto_transition BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(machine_id, from_status, to_status)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mes_machine_initial
    ON mes_status_config(machine_id) WHERE is_initial;

CREATE TABLE IF NOT EXISTS mes_andon_type (
    id BIGSERIAL PRIMARY KEY,
    type_code TEXT NOT NULL UNIQUE,
    type_name TEXT,
    category TEXT,
    description TEXT,
    require_photo BOOLEAN,
    require_confirm BOOLEAN,
    default_escalation_minutes INTEGER,
    default_process_template TEXT,
    status TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mes_andon_level (
    id BIGSERIAL PRIMARY KEY,
    level_code TEXT NOT NULL UNIQUE,
    level_name TEXT,
    level_value INTEGER NOT NULL UNIQUE,
    response_timeout_minutes INTEGER,
    handling_timeout_minutes INTEGER,
    color_code TEXT,
    description TEXT,
    status TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mes_equipment_type (
    id BIGSERIAL PRIMARY KEY,
    type_code TEXT NOT NULL UNIQUE,
    type_name TEXT,
    description TEXT,
    category TEXT,
    status_machine_id BIGINT,
    icon TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    attributes TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mes_equipment_status_config (
    id BIGSERIAL PRIMARY KEY,
    equipment_type_id BIGINT NOT NULL REFERENCES mes_equipment_type(id),
    status_code TEXT,
    status_name TEXT,
    status_category TEXT,
    is_initial BOOLEAN NOT NULL DEFAULT FALSE,
    is_final BOOLEAN NOT NULL DEFAULT FALSE,
    is_alarm BOOLEAN NOT NULL DEFAULT FALSE,
    color TEXT,
    icon TEXT,
    description TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(equipment_type_id, status_code)
);

CREATE TABLE IF NOT EXISTS mes_equipment_status_transition (
    id BIGSERIAL PRIMARY KEY,
    equipment_type_id BIGINT NOT NULL REFERENCES mes_equipment_type(id),
    from_status_code TEXT,
    to_status_code TEXT,
    transition_action TEXT,
    condition_expression TEXT,
    event_code TEXT,
    is_auto_transition BOOLEAN NOT NULL DEFAULT FALSE,
    timeout_seconds INTEGER,
    timeout_action TEXT,
    description TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(equipment_type_id, from_status_code, to_status_code),
    FOREIGN KEY(equipment_type_id, from_status_code) REFERENCES mes_equipment_status_config(equipment_type_id, status_code),
    FOREIGN KEY(equipment_type_id, to_status_code) REFERENCES mes_equipment_status_config(equipment_type_id, status_code)
);

CREATE TABLE IF NOT EXISTS mes_work_order_split_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_name TEXT,
    rule_code TEXT NOT NULL UNIQUE,
    split_type TEXT,
    description TEXT,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    split_quantity INTEGER CHECK (split_quantity > 0),
    max_child_orders INTEGER CHECK (max_child_orders BETWEEN 2 AND 1000),
    bom_id TEXT,
    process_route_id TEXT,
    conditions TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mes_andon_event (
    id BIGSERIAL PRIMARY KEY,
    event_no TEXT NOT NULL UNIQUE,
    andon_type_id BIGINT NOT NULL REFERENCES mes_andon_type(id),
    andon_type_code TEXT,
    andon_type_name TEXT,
    andon_level_id BIGINT NOT NULL REFERENCES mes_andon_level(id),
    level_code TEXT,
    level_name TEXT,
    trigger_method TEXT,
    source_system TEXT,
    source_id TEXT,
    workshop_id TEXT,
    workstation_id BIGINT,
    equipment_id TEXT,
    product_code TEXT,
    work_order_no TEXT,
    description TEXT,
    photo_url TEXT,
    reporter_id TEXT,
    reporter_name TEXT,
    status TEXT,
    current_handler_id TEXT,
    current_handler_name TEXT,
    occurred_at TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    escalation_count INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mes_work_order (
    id BIGSERIAL PRIMARY KEY,
    work_order_no TEXT NOT NULL UNIQUE,
    product_code TEXT,
    product_name TEXT,
    quantity INTEGER,
    completed_quantity INTEGER,
    status TEXT,
    status_code TEXT,
    type_code TEXT,
    priority TEXT,
    workshop_id TEXT,
    process_route_id TEXT,
    code_rule_id BIGINT,
    parent_work_order_id BIGINT,
    split_rule_id BIGINT,
    split_sequence INTEGER,
    split_type TEXT,
    planned_start_time TIMESTAMP,
    planned_end_time TIMESTAMP,
    actual_start_time TIMESTAMP,
    actual_end_time TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mes_equipment (
    id BIGSERIAL PRIMARY KEY,
    equipment_code TEXT NOT NULL UNIQUE,
    equipment_name TEXT,
    equipment_type_id BIGINT,
    equipment_type TEXT,
    workshop_id TEXT,
    workstation_id BIGINT,
    status TEXT,
    status_config_id BIGINT,
    utilization_rate NUMERIC(18,4),
    today_output INTEGER,
    current_task_no TEXT,
    last_maintenance_time TIMESTAMP,
    next_maintenance_time TIMESTAMP,
    total_running_seconds BIGINT,
    total_idle_seconds BIGINT,
    total_downtime_seconds BIGINT,
    digital_twin_device_code TEXT,
    position_x NUMERIC(18,4),
    position_y NUMERIC(18,4),
    position_z NUMERIC(18,4),
    rotation_y NUMERIC(18,4),
    ip_address TEXT,
    mac_address TEXT,
    mqtt_topic TEXT,
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE mes_andon_type ALTER COLUMN type_name SET NOT NULL;
ALTER TABLE mes_andon_type ALTER COLUMN category SET NOT NULL;
ALTER TABLE mes_andon_type ALTER COLUMN status SET NOT NULL;
ALTER TABLE mes_andon_level ALTER COLUMN level_name SET NOT NULL;
ALTER TABLE mes_andon_level ALTER COLUMN status SET NOT NULL;
ALTER TABLE mes_equipment_type ALTER COLUMN type_name SET NOT NULL;
ALTER TABLE mes_equipment_status_config ALTER COLUMN status_code SET NOT NULL;
ALTER TABLE mes_equipment_status_config ALTER COLUMN status_name SET NOT NULL;
ALTER TABLE mes_equipment_status_transition ALTER COLUMN from_status_code SET NOT NULL;
ALTER TABLE mes_equipment_status_transition ALTER COLUMN to_status_code SET NOT NULL;
ALTER TABLE mes_work_order_split_rule ALTER COLUMN rule_name SET NOT NULL;
ALTER TABLE mes_work_order_split_rule ALTER COLUMN split_type SET NOT NULL;
ALTER TABLE mes_andon_event ALTER COLUMN status SET NOT NULL;
ALTER TABLE mes_andon_event ALTER COLUMN occurred_at SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_mes_equipment_initial
    ON mes_equipment_status_config(equipment_type_id) WHERE is_initial AND is_active;
CREATE UNIQUE INDEX IF NOT EXISTS ux_mes_work_order_split_sequence
    ON mes_work_order(parent_work_order_id, split_sequence) WHERE parent_work_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_mes_andon_status_time ON mes_andon_event(status, occurred_at);
CREATE INDEX IF NOT EXISTS ix_mes_andon_workshop ON mes_andon_event(workshop_id);
CREATE INDEX IF NOT EXISTS ix_mes_andon_type ON mes_andon_event(andon_type_id);
CREATE INDEX IF NOT EXISTS ix_mes_andon_level ON mes_andon_event(andon_level_id);
CREATE INDEX IF NOT EXISTS ix_mes_andon_equipment ON mes_andon_event(equipment_id);
CREATE INDEX IF NOT EXISTS ix_mes_equipment_status_from ON mes_equipment_status_transition(equipment_type_id, from_status_code);
CREATE INDEX IF NOT EXISTS ix_mes_split_rule_enabled ON mes_work_order_split_rule(enabled, split_type);
COMMIT;