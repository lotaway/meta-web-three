-- schema.sql
-- db/migration/V*.sql “”

-- =====================
-- V1: process flow init
-- =====================
-- Flowable

CREATE TABLE IF NOT EXISTS mes_process_flow_template (
    id BIGINT PRIMARY KEY,
    template_code VARCHAR(64) NOT NULL UNIQUE,
    template_name VARCHAR(128) NOT NULL,
    description VARCHAR(512),
    version INT DEFAULT 1,
 flow_data TEXT, -- JSON,
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PUBLISHED/ARCHIVED,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS mes_process_node_type (
    id BIGINT PRIMARY KEY,
    node_type_code VARCHAR(64) NOT NULL UNIQUE,
    node_type_name VARCHAR(128) NOT NULL,
    category VARCHAR(32) NOT NULL, -- START/END/TASK/GATEWAY/SERVICE/EQUIPMENT/SUB_PROCESS,
    icon VARCHAR(32),
    config_schema TEXT, -- JSON Schema for node configuration,
    description VARCHAR(512),
    enabled BOOLEAN DEFAULT TRUE,
    sort_order INT DEFAULT 0,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS mes_process_flow_instance (
    id BIGINT PRIMARY KEY,
    instance_code VARCHAR(64) NOT NULL UNIQUE,
    template_id BIGINT NOT NULL,
    template_name VARCHAR(128),
    business_type VARCHAR(64), -- WORK_ORDER/QC/MAINTENANCE/ANDON etc.,
    business_key VARCHAR(128), -- Reference to business entity,
    current_node_id VARCHAR(64),
    current_node_name VARCHAR(128),
    status VARCHAR(32) DEFAULT 'RUNNING', -- RUNNING/SUSPENDED/COMPLETED/TERMINATED,
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
    FOREIGN KEY (template_id) REFERENCES mes_process_flow_template (id)
);

CREATE INDEX idx_template_status ON mes_process_flow_template (status);

CREATE INDEX idx_template_code ON mes_process_flow_template (template_code);

CREATE INDEX idx_instance_code ON mes_process_flow_instance (instance_code);

CREATE INDEX idx_instance_business ON mes_process_flow_instance (business_type, business_key);

CREATE INDEX idx_instance_status ON mes_process_flow_instance (status);

INSERT INTO
    mes_process_node_type (
        id,
        node_type_code,
        node_type_name,
        category,
        icon,
        sort_order,
        description,
        enabled,
        deleted
)
VALUES (
        1,
        'start',
        'Start',
        'START',
        '▶',
        1,
        'Process start node',
        TRUE,
        FALSE
    ),
    (
        2,
        'end',
        'End',
        'END',
        '■',
        2,
        'Process end node',
        TRUE,
        FALSE
    ),
    (
        3,
        'manual_task',
        'Manual Task',
        'TASK',
        '👤',
        3,
        'Manual operation task node',
        TRUE,
        FALSE
    ),
    (
        4,
        'equipment_task',
        'Equipment Task',
        'EQUIPMENT',
        '⚙',
        4,
        'Equipment operation task node',
        TRUE,
        FALSE
    ),
    (
        5,
        'data_process',
        'Data Process',
        'SERVICE',
        '⚡',
        5,
        'Data processing service node',
        TRUE,
        FALSE
    ),
    (
        6,
        'system_integration',
        'System Integration',
        'SERVICE',
        '🔗',
        6,
        'System integration service node',
        TRUE,
        FALSE
    ),
    (
        7,
        'exclusive_gateway',
        'Exclusive Gateway',
        'GATEWAY',
        '◇',
        7,
        'Exclusive choice gateway node',
        TRUE,
        FALSE
    ),
    (
        8,
        'parallel_gateway',
        'Parallel Gateway',
        'GATEWAY',
        '◆',
        8,
        'Parallel fork gateway node',
        TRUE,
        FALSE
    ),
    (
        9,
        'inclusive_gateway',
        'Inclusive Gateway',
        'GATEWAY',
        '◈',
        9,
        'Inclusive merge gateway node',
        TRUE,
        FALSE
    ),
    (
        10,
        'sub_process',
        'Sub Process',
        'SUB_PROCESS',
        '🔄',
        10,
        'Nested sub-process node',
        TRUE,
        FALSE
)
ON CONFLICT (node_type_code) DO NOTHING;

-- =====================
-- V2: process flow template version
-- =====================
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
    FOREIGN KEY (template_id) REFERENCES mes_process_flow_template (id)
);

CREATE INDEX idx_version_template ON mes_process_flow_template_version (template_id);

CREATE INDEX idx_version_template_version ON mes_process_flow_template_version (template_id, version);

-- =====================
-- V3: pokayoke rule
-- =====================
CREATE TABLE IF NOT EXISTS mes_pokayoke_rule (
    id BIGINT PRIMARY KEY,
    rule_code VARCHAR(64) NOT NULL UNIQUE,
    rule_name VARCHAR(128) NOT NULL,
    rule_type VARCHAR(32) NOT NULL, -- MATERIAL_CHECK/SEQUENCE_CHECK/PARAMETER_CHECK/QUALITY_CHECK,
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/ACTIVE/INACTIVE,
    workstation_id BIGINT,
    process_code VARCHAR(64),
    product_code VARCHAR(64),
    trigger_condition VARCHAR(64), -- ON_MATERIAL_SCAN/ON_TASK_START/ON_TASK_COMPLETE/ON_PARAMETER_RECORD/MANUAL_TRIGGER,
    actions_json TEXT, -- JSON array of CheckAction,
    priority INT DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    description VARCHAR(512),
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_pokayoke_status ON mes_pokayoke_rule (status);

CREATE INDEX idx_pokayoke_workstation ON mes_pokayoke_rule (workstation_id);

CREATE INDEX idx_pokayoke_process ON mes_pokayoke_rule (process_code);

-- =====================
-- V4: report designer
-- =====================

CREATE TABLE IF NOT EXISTS mes_report_template (
    id BIGINT PRIMARY KEY,
    template_code VARCHAR(64) NOT NULL UNIQUE,
    template_name VARCHAR(128) NOT NULL,
    report_type VARCHAR(32) NOT NULL, -- LIST/CROSS/CHART/GROUP,
    description VARCHAR(512),
 config_json TEXT,    datasource_type VARCHAR(32), -- MES/EXTERNAL/API,
    datasource_config TEXT,
 query_sql TEXT, -- SQL,
    parameters_json TEXT,
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PUBLISHED/ARCHIVED,
    version INT DEFAULT 1,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS mes_report_datasource (
    id BIGINT PRIMARY KEY,
    datasource_code VARCHAR(64) NOT NULL UNIQUE,
    datasource_name VARCHAR(128) NOT NULL,
    datasource_type VARCHAR(32) NOT NULL, -- MES/ORACLE/MYSQL/POSTGRESQL/REST_API,
 connection_config TEXT, -- JSON,
    description VARCHAR(512),
    enabled BOOLEAN DEFAULT TRUE,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_report_template_status ON mes_report_template (status);

CREATE INDEX idx_report_template_type ON mes_report_template (report_type);

CREATE INDEX idx_report_datasource_type ON mes_report_datasource (datasource_type);

-- =====================
-- V5: dashboard designer
-- =====================
-- /

CREATE TABLE IF NOT EXISTS mes_dashboard_template (
    id BIGINT PRIMARY KEY,
    template_code VARCHAR(64) NOT NULL UNIQUE,
    template_name VARCHAR(128) NOT NULL,
    template_type VARCHAR(32) NOT NULL, -- PRODUCTION/QUALITY/EQUIPMENT/OEE,
    description VARCHAR(512),
    layout_json TEXT,
 components_json TEXT, -- JSON,
    datasource_config TEXT,
    refresh_interval INT DEFAULT 30,
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PUBLISHED/ARCHIVED,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS mes_dashboard_component (
    id BIGINT PRIMARY KEY,
    component_code VARCHAR(64) NOT NULL UNIQUE,
    component_name VARCHAR(128) NOT NULL,
    component_type VARCHAR(32) NOT NULL, -- CHART/TABLE/DIAGRAM/INDICATOR,
 config_schema TEXT, -- Schema,
    default_config TEXT,
    icon VARCHAR(32),
    description VARCHAR(512),
    enabled BOOLEAN DEFAULT TRUE,
    sort_order INT DEFAULT 0,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_dashboard_template_status ON mes_dashboard_template (status);

CREATE INDEX idx_dashboard_template_type ON mes_dashboard_template (template_type);

CREATE INDEX idx_dashboard_component_type ON mes_dashboard_component (component_type);

-- =====================
-- V6: defect code
-- =====================
CREATE TABLE IF NOT EXISTS mes_qc_defect_code (
    id BIGSERIAL PRIMARY KEY,
    defect_code VARCHAR(50) NOT NULL UNIQUE,
    defect_name VARCHAR(100) NOT NULL,
 category VARCHAR(30) NOT NULL, -- : DIMENSIONAL, SURFACE, MATERIAL, ASSEMBLY, ELECTRICAL, FUNCTIONAL, PACKAGING, OTHER,
 severity VARCHAR(20) NOT NULL, -- : CRITICAL, MAJOR, MINOR,
    is_critical BOOLEAN DEFAULT FALSE,
    description VARCHAR(500),
    disposition_guide VARCHAR(500),
    is_enabled BOOLEAN DEFAULT TRUE,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_defect_code_code ON mes_qc_defect_code (defect_code);

CREATE INDEX idx_defect_code_category ON mes_qc_defect_code (category);

CREATE INDEX idx_defect_code_severity ON mes_qc_defect_code (severity);

CREATE INDEX idx_defect_code_enabled ON mes_qc_defect_code (is_enabled);

-- =====================
-- V7: qc trigger rule
-- =====================
-- V7: Qc Trigger Rule
CREATE TABLE IF NOT EXISTS mes_qc_trigger_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_code VARCHAR(50) NOT NULL UNIQUE,  -- Rule Code,
    rule_name VARCHAR(100) NOT NULL,  -- Rule Name,
    trigger_type VARCHAR(20) NOT NULL,  -- Trigger Type: BY_BATCH, BY_TIME, BY_QUANTITY, BY_EVENT, MANUAL,
    target_object VARCHAR(100),  -- Target Object: WORK_ORDER, PRODUCTION_TASK, PROCESS_STEP,
    condition_json TEXT,  -- Trigger Condition JSON,
    inspection_type VARCHAR(50),  -- Inspection Type,
    inspection_plan_code VARCHAR(50),  -- Inspection Plan Code,
    is_enabled BOOLEAN DEFAULT TRUE,  -- Is Enabled,
    priority INT DEFAULT 0,  -- Priority,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Created At,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Updated At
);
CREATE INDEX IF NOT EXISTS idx_rule_code ON mes_qc_trigger_rule (rule_code);
CREATE INDEX IF NOT EXISTS idx_trigger_type ON mes_qc_trigger_rule (trigger_type);
CREATE INDEX IF NOT EXISTS idx_is_enabled ON mes_qc_trigger_rule (is_enabled);

-- =====================
-- V8: non conformance disposition
-- =====================
-- V8: Non-Conformance Disposition
CREATE TABLE IF NOT EXISTS mes_qc_non_conformance (
    id BIGSERIAL PRIMARY KEY,
    disposition_code VARCHAR(50) NOT NULL UNIQUE,  -- Disposition Code,
    disposition_name VARCHAR(100) NOT NULL,  -- Disposition Name,
    type VARCHAR(20) NOT NULL,  -- Type: SCRAP, REWORK, REPAIR, RETURN, USE_AS_IS, SPECIAL_ACCEPTANCE,
    steps_json TEXT,  -- Steps JSON,
    is_enabled BOOLEAN DEFAULT TRUE,  -- Is Enabled,
    sort_order INT DEFAULT 0,  -- Sort Order,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Created At,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Updated At
);
CREATE INDEX IF NOT EXISTS idx_disposition_code ON mes_qc_non_conformance (disposition_code);
CREATE INDEX IF NOT EXISTS idx_type ON mes_qc_non_conformance (type);
CREATE INDEX IF NOT EXISTS idx_is_enabled ON mes_qc_non_conformance (is_enabled);

-- =====================
-- V9: spc control chart
-- =====================
-- V9: SPC Control Chart (SPC)
CREATE TABLE IF NOT EXISTS mes_qc_spc_control_chart (
    id BIGSERIAL PRIMARY KEY,
    chart_code VARCHAR(50) NOT NULL UNIQUE,  -- Chart Code,
    chart_name VARCHAR(100) NOT NULL,  -- Chart Name,
    chart_type VARCHAR(20) NOT NULL,  -- Chart Type: XBAR_R, XBAR_S, X_MR, P_CHART, NP_CHART, C_CHART, CU_CHART,
    parameter_code VARCHAR(50),  -- Parameter Code,
    limits_json TEXT,  -- Control Limits JSON,
    alarm_rules_json TEXT,  -- Alarm Rules JSON,
    sampling_config_json TEXT,  -- Sampling Config JSON,
    is_enabled BOOLEAN DEFAULT TRUE,  -- Is Enabled,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Created At,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Updated At
);
CREATE INDEX IF NOT EXISTS idx_chart_code ON mes_qc_spc_control_chart (chart_code);
CREATE INDEX IF NOT EXISTS idx_chart_type ON mes_qc_spc_control_chart (chart_type);
CREATE INDEX IF NOT EXISTS idx_parameter_code ON mes_qc_spc_control_chart (parameter_code);
CREATE INDEX IF NOT EXISTS idx_is_enabled ON mes_qc_spc_control_chart (is_enabled);

-- =====================
-- V10: work report
-- =====================
-- V10: Work Report
CREATE TABLE IF NOT EXISTS mes_work_report (
    id BIGSERIAL PRIMARY KEY,
    report_no VARCHAR(50) NOT NULL UNIQUE,  -- Report No,
    task_id BIGINT,  -- Production Task ID,
    task_no VARCHAR(50),  -- Task No,
    work_order_id BIGINT,  -- Work Order ID,
    work_order_no VARCHAR(50),  -- Work Order No,
    workstation_id BIGINT,  -- Workstation ID,
    workstation_name VARCHAR(100),  -- Workstation Name,
    process_code VARCHAR(50),  -- Process Code,
    process_name VARCHAR(100),  -- Process Name,
    step_no INT,  -- Step No,
    operator_id VARCHAR(50),  -- Operator ID,
    operator_name VARCHAR(100),  -- Operator Name,
    report_time TIMESTAMP,  -- Report Time,
    quantity INT,  -- Total Quantity,
    qualified_quantity INT,  -- Qualified Quantity,
    defective_quantity INT,  -- Defective Quantity,
    duration_minutes INT,  -- Duration in Minutes,
    parameter_values_json TEXT,  -- Parameter Values JSON,
    remarks VARCHAR(500),  -- Remarks,
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',  -- Status: DRAFT, SUBMITTED, QUALITY_CHECKED, CONFIRMED, CANCELLED,
    created_by VARCHAR(50),  -- Created By,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Created At,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Updated At
);
CREATE INDEX IF NOT EXISTS idx_report_no ON mes_work_report (report_no);
CREATE INDEX IF NOT EXISTS idx_task_id ON mes_work_report (task_id);
CREATE INDEX IF NOT EXISTS idx_work_order_id ON mes_work_report (work_order_id);
CREATE INDEX IF NOT EXISTS idx_workstation_id ON mes_work_report (workstation_id);
CREATE INDEX IF NOT EXISTS idx_operator_id ON mes_work_report (operator_id);
CREATE INDEX IF NOT EXISTS idx_status ON mes_work_report (status);
CREATE INDEX IF NOT EXISTS idx_report_time ON mes_work_report (report_time);

-- =====================
-- V11: workstation
-- =====================
-- Workstation  table
CREATE TABLE IF NOT EXISTS mes_workstation (
    id BIGSERIAL PRIMARY KEY,
    workstation_code VARCHAR(50) NOT NULL UNIQUE,
    workstation_name VARCHAR(100) NOT NULL,
 workshop_id VARCHAR(50), -- ID,
    workshop_name VARCHAR(100),
 type VARCHAR(50) NOT NULL, -- : ASSEMBLY/INSPECTION/PACKAGING/STORAGE/MATERIAL_PREP/TESTING/REWORK/OTHER,
 status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE', -- : ACTIVE/INACTIVE/MAINTENANCE/FAULT,
    location VARCHAR(200),
 capacity INT, -- /,
    description VARCHAR(500),
 equipment_ids TEXT, -- ID JSON,
 equipment_codes TEXT, -- JSON,
 tool_ids TEXT, -- ID JSON,
 tool_names TEXT, -- JSON,
 operator_ids TEXT, -- ID JSON,
 operator_names TEXT, -- JSON,
 extension_fields TEXT, -- JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_workstation_code ON mes_workstation (workstation_code);
CREATE INDEX IF NOT EXISTS idx_workshop_id ON mes_workstation (workshop_id);
CREATE INDEX IF NOT EXISTS idx_type ON mes_workstation (type);
CREATE INDEX IF NOT EXISTS idx_status ON mes_workstation (status);

-- =====================
-- V12: parameter_group_template & process_parameter
-- =====================
-- Parameter Group Template  table
CREATE TABLE IF NOT EXISTS mes_parameter_group_template (
    id BIGSERIAL PRIMARY KEY,
    template_code VARCHAR(50) NOT NULL UNIQUE,
    template_name VARCHAR(100) NOT NULL,
    product_type VARCHAR(50),
    description VARCHAR(500),
 status VARCHAR(50) NOT NULL DEFAULT 'DRAFT', -- : DRAFT/ACTIVE/INACTIVE/ARCHIVED,
    display_order INT DEFAULT 0,
 parameter_ids TEXT, -- ID JSON,
 parameter_codes TEXT, -- JSON,
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_template_code ON mes_parameter_group_template (template_code);
CREATE INDEX IF NOT EXISTS idx_product_type ON mes_parameter_group_template (product_type);
CREATE INDEX IF NOT EXISTS idx_status ON mes_parameter_group_template (status);

-- Process Parameter  table
CREATE TABLE IF NOT EXISTS mes_process_parameter (
    id BIGSERIAL PRIMARY KEY,
    param_code VARCHAR(50) NOT NULL UNIQUE,
    param_name VARCHAR(100) NOT NULL,
 route_id BIGINT, -- ID,
    route_code VARCHAR(50),
    step_no INT,
    step_code VARCHAR(50),
 param_type VARCHAR(50), -- : TEMPERATURE/PRESSURE/SPEED/TIME/CURRENT/VOLTAGE/FORCE/LENGTH/ANGLE/WEIGHT/VOLUME/SPEED_PER_MINUTE/HUMIDITY/QUALITY/COUNT/OTHER,
 data_type VARCHAR(50), -- : INTEGER/DECIMAL/TEXT/BOOLEAN,
    unit VARCHAR(20),
    standard_value DECIMAL(18,4),
    upper_limit DECIMAL(18,4),
    lower_limit DECIMAL(18,4),
 collection_method VARCHAR(50), -- : MANUAL/AUTO_SENSOR/PLC/BARCODE,
    device_address VARCHAR(100),
    is_required SMALLINT DEFAULT 0,
    validation_rule VARCHAR(200),
    alarm_threshold DECIMAL(18,4),
    display_order INT DEFAULT 0,
    param_group VARCHAR(50),
    remark VARCHAR(500),
 status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE', -- : ACTIVE/INACTIVE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_param_code ON mes_process_parameter (param_code);
CREATE INDEX IF NOT EXISTS idx_route_id ON mes_process_parameter (route_id);
CREATE INDEX IF NOT EXISTS idx_route_code ON mes_process_parameter (route_code);
CREATE INDEX IF NOT EXISTS idx_step_no ON mes_process_parameter (step_no);
CREATE INDEX IF NOT EXISTS idx_param_type ON mes_process_parameter (param_type);
CREATE INDEX IF NOT EXISTS idx_status ON mes_process_parameter (status);

-- =====================
-- V13: process_route
-- =====================
-- Process Route  table
CREATE TABLE IF NOT EXISTS mes_process_route (
    id BIGSERIAL PRIMARY KEY,
    route_code VARCHAR(50) NOT NULL UNIQUE,
    route_name VARCHAR(100) NOT NULL,
    product_code VARCHAR(50),
    version INT NOT NULL DEFAULT 1,
 status VARCHAR(50) NOT NULL DEFAULT 'DRAFT', -- : DRAFT/ACTIVE/ARCHIVED,
    effective_date TIMESTAMP,
    expiry_date TIMESTAMP,
 steps TEXT, -- JSON ,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_route_code ON mes_process_route (route_code);
CREATE INDEX IF NOT EXISTS idx_product_code ON mes_process_route (product_code);
CREATE INDEX IF NOT EXISTS idx_version ON mes_process_route (version);
CREATE INDEX IF NOT EXISTS idx_status ON mes_process_route (status);
CREATE INDEX IF NOT EXISTS idx_effective_expiry ON mes_process_route (effective_date, expiry_date);

-- =====================
-- V14: scheduling
-- =====================
CREATE TABLE IF NOT EXISTS mes_schedule_order (
    id BIGSERIAL PRIMARY KEY,
    schedule_no VARCHAR(64) NOT NULL UNIQUE,
    order_no VARCHAR(64) NOT NULL,
    product_code VARCHAR(64),
    product_name VARCHAR(128),
    quantity DECIMAL(18, 4) DEFAULT 0,
    completed_quantity DECIMAL(18, 4) DEFAULT 0,
    due_date TIMESTAMP,
    scheduled_start_time TIMESTAMP,
    scheduled_end_time TIMESTAMP,
    actual_start_time TIMESTAMP,
    actual_end_time TIMESTAMP,
 priority VARCHAR(32) DEFAULT 'NORMAL', -- : LOW/NORMAL/HIGH/URGENT,
 status VARCHAR(32) DEFAULT 'PENDING', -- : PENDING/SCHEDULED/IN_PROGRESS/COMPLETED/DELAYED/CANCELLED,
 workshop_id VARCHAR(64), -- ID,
    route_code VARCHAR(64),
    remark VARCHAR(512),
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_schedule_no ON mes_schedule_order (schedule_no);
CREATE INDEX IF NOT EXISTS idx_order_no ON mes_schedule_order (order_no);
CREATE INDEX IF NOT EXISTS idx_workshop_id ON mes_schedule_order (workshop_id);
CREATE INDEX IF NOT EXISTS idx_status ON mes_schedule_order (status);
CREATE INDEX IF NOT EXISTS idx_priority ON mes_schedule_order (priority);
CREATE INDEX IF NOT EXISTS idx_due_date ON mes_schedule_order (due_date);

CREATE TABLE IF NOT EXISTS mes_schedule_operation (
    id BIGSERIAL PRIMARY KEY,
 schedule_order_id BIGINT NOT NULL, -- ID,
    operation_code VARCHAR(64) NOT NULL,
    operation_name VARCHAR(128),
    sequence_no INT DEFAULT 1,
    resource_code VARCHAR(64),
    resource_name VARCHAR(128),
    setup_time_minutes DECIMAL(12, 2) DEFAULT 0,
    processing_time_minutes DECIMAL(12, 2) DEFAULT 0,
    teardown_time_minutes DECIMAL(12, 2) DEFAULT 0,
 status VARCHAR(32) DEFAULT 'PENDING', -- : PENDING/SCHEDULED/IN_PROGRESS/COMPLETED/BLOCKED,
    scheduled_start_time TIMESTAMP,
    scheduled_end_time TIMESTAMP,
    FOREIGN KEY (schedule_order_id) REFERENCES mes_schedule_order (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_schedule_order_id ON mes_schedule_operation (schedule_order_id);
CREATE INDEX IF NOT EXISTS idx_operation_code ON mes_schedule_operation (operation_code);
CREATE INDEX IF NOT EXISTS idx_resource_code ON mes_schedule_operation (resource_code);
CREATE INDEX IF NOT EXISTS idx_status ON mes_schedule_operation (status);

CREATE TABLE IF NOT EXISTS mes_schedule_resource (
    id BIGSERIAL PRIMARY KEY,
    resource_code VARCHAR(64) NOT NULL UNIQUE,
    resource_name VARCHAR(128) NOT NULL,
 resource_type VARCHAR(32) NOT NULL, -- : EQUIPMENT/WORK_CENTER/LABOR/TOOL,
 status VARCHAR(32) DEFAULT 'AVAILABLE', -- : AVAILABLE/OCCUPIED/MAINTENANCE/OFFLINE,
 workshop_id VARCHAR(64), -- ID,
    capacity_per_shift DOUBLE PRECISION,
    calendar_code VARCHAR(64),
    description VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_resource_code ON mes_schedule_resource (resource_code);
CREATE INDEX IF NOT EXISTS idx_resource_type ON mes_schedule_resource (resource_type);
CREATE INDEX IF NOT EXISTS idx_workshop_id ON mes_schedule_resource (workshop_id);
CREATE INDEX IF NOT EXISTS idx_status ON mes_schedule_resource (status);

-- =====================
-- V15: labor & time tracking
-- =====================
CREATE TABLE IF NOT EXISTS mes_operator (
    id BIGSERIAL PRIMARY KEY,
    operator_code VARCHAR(64) NOT NULL UNIQUE,
    operator_name VARCHAR(128) NOT NULL,
    department VARCHAR(64),
    job_title VARCHAR(64),
    shift_group VARCHAR(32),
 status VARCHAR(32) DEFAULT 'ACTIVE', -- : ACTIVE/INACTIVE/ON_LEAVE/TERMINATED,
    phone VARCHAR(32),
    email VARCHAR(128),
    id_card_no VARCHAR(32),
    hire_date TIMESTAMP,
    remark VARCHAR(512),
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_operator_code ON mes_operator (operator_code);
CREATE INDEX IF NOT EXISTS idx_department ON mes_operator (department);
CREATE INDEX IF NOT EXISTS idx_status ON mes_operator (status);

CREATE TABLE IF NOT EXISTS mes_operator_skill (
    id BIGSERIAL PRIMARY KEY,
 operator_id BIGINT NOT NULL, -- ID,
    skill_code VARCHAR(64) NOT NULL,
    skill_name VARCHAR(128),
 skill_level VARCHAR(32) DEFAULT 'TRAINEE', -- : TRAINEE/JUNIOR/MIDDLE/SENIOR/MASTER,
    certified BOOLEAN DEFAULT FALSE,
    certified_at TIMESTAMP,
    expiry_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_operator_id ON mes_operator_skill (operator_id);
CREATE INDEX IF NOT EXISTS idx_skill_code ON mes_operator_skill (skill_code);

CREATE TABLE IF NOT EXISTS mes_work_center_assignment (
    id BIGSERIAL PRIMARY KEY,
 operator_id BIGINT NOT NULL, -- ID,
 work_center_id VARCHAR(64) NOT NULL, -- ID,
    work_center_name VARCHAR(128),
    start_date DATE,
    end_date DATE,
 shift_type VARCHAR(32) DEFAULT 'DAY', -- : DAY/NIGHT/MIDDLE/ROTATING,
 status VARCHAR(32) DEFAULT 'ACTIVE', -- : ACTIVE/INACTIVE,
    remark VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_operator_id ON mes_work_center_assignment (operator_id);
CREATE INDEX IF NOT EXISTS idx_work_center_id ON mes_work_center_assignment (work_center_id);
CREATE INDEX IF NOT EXISTS idx_status ON mes_work_center_assignment (status);

CREATE TABLE IF NOT EXISTS mes_time_record (
    id BIGSERIAL PRIMARY KEY,
 operator_id BIGINT NOT NULL, -- ID,
    operator_code VARCHAR(64),
    operator_name VARCHAR(128),
    work_order_no VARCHAR(64),
    task_no VARCHAR(64),
    operation_code VARCHAR(64),
 work_center_id VARCHAR(64), -- ID,
    record_date DATE,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_hours DECIMAL(12, 2) DEFAULT 0,
 record_type VARCHAR(32) DEFAULT 'REGULAR', -- : REGULAR/OVERTIME/VACATION/SICK,
 status VARCHAR(32) DEFAULT 'DRAFT', -- : DRAFT/SUBMITTED/APPROVED/REJECTED,
    approved_by VARCHAR(64),
    approved_at TIMESTAMP,
    remark VARCHAR(512),
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_operator_id ON mes_time_record (operator_id);
CREATE INDEX IF NOT EXISTS idx_record_date ON mes_time_record (record_date);
CREATE INDEX IF NOT EXISTS idx_work_order_no ON mes_time_record (work_order_no);
CREATE INDEX IF NOT EXISTS idx_status ON mes_time_record (status);

CREATE TABLE IF NOT EXISTS mes_attendance (
    id BIGSERIAL PRIMARY KEY,
 operator_id BIGINT NOT NULL, -- ID,
    operator_code VARCHAR(64),
    operator_name VARCHAR(128),
    attendance_date DATE,
    clock_in TIME,
    clock_out TIME,
    scheduled_start TIME,
    scheduled_end TIME,
 status VARCHAR(32) DEFAULT 'ABSENT', -- : PRESENT/LATE/ABSENT/HALF_DAY/OVERTIME/VACATION/SICK/BUSINESS_TRIP,
    overtime BOOLEAN DEFAULT FALSE,
    remark VARCHAR(512),
    created_by VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_operator_id ON mes_attendance (operator_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON mes_attendance (attendance_date);
CREATE INDEX IF NOT EXISTS idx_status ON mes_attendance (status);
