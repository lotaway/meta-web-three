-- schema.sql
-- 将 db/migration/V*.sql 的表结构与初始化数据合并后的“单一初始化脚本”
-- 当前项目不再依赖逐步升级迁移方式（只用本脚本完成初始化）

-- =====================
-- V1: process flow init
-- =====================
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
    FOREIGN KEY (template_id) REFERENCES mes_process_flow_template (id)
);

-- 创建索引
CREATE INDEX idx_template_status ON mes_process_flow_template (status);

CREATE INDEX idx_template_code ON mes_process_flow_template (template_code);

CREATE INDEX idx_instance_code ON mes_process_flow_instance (instance_code);

CREATE INDEX idx_instance_business ON mes_process_flow_instance (business_type, business_key);

CREATE INDEX idx_instance_status ON mes_process_flow_instance (status);

-- 初始化预置节点类型
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
        '开始',
        'START',
        '▶',
        1,
        '流程开始节点',
        TRUE,
        FALSE
    ),
    (
        2,
        'end',
        '结束',
        'END',
        '■',
        2,
        '流程结束节点',
        TRUE,
        FALSE
    ),
    (
        3,
        'manual_task',
        '人工任务',
        'TASK',
        '👤',
        3,
        '需要人工处理的任务节点',
        TRUE,
        FALSE
    ),
    (
        4,
        'equipment_task',
        '设备交互',
        'EQUIPMENT',
        '⚙',
        4,
        '与设备进行数据交互的节点',
        TRUE,
        FALSE
    ),
    (
        5,
        'data_process',
        '数据处理',
        'SERVICE',
        '⚡',
        5,
        '数据处理和服务调用节点',
        TRUE,
        FALSE
    ),
    (
        6,
        'system_integration',
        '系统集成',
        'SERVICE',
        '🔗',
        6,
        '与外部系统集成的节点',
        TRUE,
        FALSE
    ),
    (
        7,
        'exclusive_gateway',
        '排他网关',
        'GATEWAY',
        '◇',
        7,
        '根据条件选择分支',
        TRUE,
        FALSE
    ),
    (
        8,
        'parallel_gateway',
        '并行网关',
        'GATEWAY',
        '◆',
        8,
        '并行执行多个分支',
        TRUE,
        FALSE
    ),
    (
        9,
        'inclusive_gateway',
        '包容网关',
        'GATEWAY',
        '◈',
        9,
        '根据条件并行执行分支',
        TRUE,
        FALSE
    ),
    (
        10,
        'sub_process',
        '子流程',
        'SUB_PROCESS',
        '🔄',
        10,
        '调用子流程',
        TRUE,
        FALSE
    )
ON CONFLICT (node_type_code) DO NOTHING;

-- =====================
-- V2: process flow template version
-- =====================
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
    FOREIGN KEY (template_id) REFERENCES mes_process_flow_template (id)
);

CREATE INDEX idx_version_template ON mes_process_flow_template_version (template_id);

CREATE INDEX idx_version_template_version ON mes_process_flow_template_version (template_id, version);

-- =====================
-- V3: pokayoke rule
-- =====================
-- 防错规则引擎表
CREATE TABLE IF NOT EXISTS mes_pokayoke_rule (
    id BIGINT PRIMARY KEY,
    rule_code VARCHAR(64) NOT NULL UNIQUE,
    rule_name VARCHAR(128) NOT NULL,
    rule_type VARCHAR(32) NOT NULL, -- MATERIAL_CHECK/SEQUENCE_CHECK/PARAMETER_CHECK/QUALITY_CHECK
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/ACTIVE/INACTIVE
    workstation_id BIGINT,
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

CREATE INDEX idx_pokayoke_status ON mes_pokayoke_rule (status);

CREATE INDEX idx_pokayoke_workstation ON mes_pokayoke_rule (workstation_id);

CREATE INDEX idx_pokayoke_process ON mes_pokayoke_rule (process_code);

-- =====================
-- V4: report designer
-- =====================
-- 报表设计器表

-- 报表模板表
CREATE TABLE IF NOT EXISTS mes_report_template (
    id BIGINT PRIMARY KEY,
    template_code VARCHAR(64) NOT NULL UNIQUE,
    template_name VARCHAR(128) NOT NULL,
    report_type VARCHAR(32) NOT NULL, -- LIST/CROSS/CHART/GROUP
    description VARCHAR(512),
    config_json TEXT, -- 报表布局、列定义、分组等配置
    datasource_type VARCHAR(32), -- MES/EXTERNAL/API
    datasource_config TEXT, -- 数据源连接配置
    query_sql TEXT, -- 自定义SQL查询
    parameters_json TEXT, -- 查询参数定义
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PUBLISHED/ARCHIVED
    version INT DEFAULT 1,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

-- 数据源配置表
CREATE TABLE IF NOT EXISTS mes_report_datasource (
    id BIGINT PRIMARY KEY,
    datasource_code VARCHAR(64) NOT NULL UNIQUE,
    datasource_name VARCHAR(128) NOT NULL,
    datasource_type VARCHAR(32) NOT NULL, -- MES/ORACLE/MYSQL/POSTGRESQL/REST_API
    connection_config TEXT, -- JSON格式的连接配置
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
-- 看板/大屏配置表

-- 看板模板表
CREATE TABLE IF NOT EXISTS mes_dashboard_template (
    id BIGINT PRIMARY KEY,
    template_code VARCHAR(64) NOT NULL UNIQUE,
    template_name VARCHAR(128) NOT NULL,
    template_type VARCHAR(32) NOT NULL, -- PRODUCTION/QUALITY/EQUIPMENT/OEE
    description VARCHAR(512),
    layout_json TEXT, -- 拖拽式布局配置
    components_json TEXT, -- 组件配置JSON
    datasource_config TEXT, -- 数据源配置
    refresh_interval INT DEFAULT 30, -- 刷新间隔（秒）
    status VARCHAR(32) DEFAULT 'DRAFT', -- DRAFT/PUBLISHED/ARCHIVED
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by BIGINT,
    updated_at TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

-- 可视化组件库表
CREATE TABLE IF NOT EXISTS mes_dashboard_component (
    id BIGINT PRIMARY KEY,
    component_code VARCHAR(64) NOT NULL UNIQUE,
    component_name VARCHAR(128) NOT NULL,
    component_type VARCHAR(32) NOT NULL, -- CHART/TABLE/DIAGRAM/INDICATOR
    config_schema TEXT, -- 组件配置Schema
    default_config TEXT, -- 默认配置
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
-- 缺陷代码表
CREATE TABLE IF NOT EXISTS mes_qc_defect_code (
    id BIGSERIAL PRIMARY KEY,
    defect_code VARCHAR(50) NOT NULL UNIQUE COMMENT '缺陷代码',
    defect_name VARCHAR(100) NOT NULL COMMENT '缺陷名称',
    category VARCHAR(30) NOT NULL COMMENT '缺陷分类: DIMENSIONAL, SURFACE, MATERIAL, ASSEMBLY, ELECTRICAL, FUNCTIONAL, PACKAGING, OTHER',
    severity VARCHAR(20) NOT NULL COMMENT '严重等级: CRITICAL, MAJOR, MINOR',
    is_critical BOOLEAN DEFAULT FALSE COMMENT '是否致命',
    description VARCHAR(500) COMMENT '缺陷描述',
    disposition_guide VARCHAR(500) COMMENT '处置指南',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
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
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = 'QC Trigger Rule';

-- =====================
-- V8: non conformance disposition
-- =====================
-- V8: Non-Conformance Disposition (不合格处置流程)
CREATE TABLE IF NOT EXISTS mes_qc_non_conformance (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    disposition_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'Disposition Code',
    disposition_name VARCHAR(100) NOT NULL COMMENT 'Disposition Name',
    type VARCHAR(20) NOT NULL COMMENT 'Type: SCRAP, REWORK, REPAIR, RETURN, USE_AS_IS, SPECIAL_ACCEPTANCE',
    steps_json TEXT COMMENT 'Steps JSON',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT 'Is Enabled',
    sort_order INT DEFAULT 0 COMMENT 'Sort Order',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated At',
    INDEX idx_disposition_code (disposition_code),
    INDEX idx_type (type),
    INDEX idx_is_enabled (is_enabled)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = 'QC Non-Conformance Disposition';

-- =====================
-- V9: spc control chart
-- =====================
-- V9: SPC Control Chart (SPC控制图)
CREATE TABLE IF NOT EXISTS mes_qc_spc_control_chart (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chart_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'Chart Code',
    chart_name VARCHAR(100) NOT NULL COMMENT 'Chart Name',
    chart_type VARCHAR(20) NOT NULL COMMENT 'Chart Type: XBAR_R, XBAR_S, X_MR, P_CHART, NP_CHART, C_CHART, CU_CHART',
    parameter_code VARCHAR(50) COMMENT 'Parameter Code',
    limits_json TEXT COMMENT 'Control Limits JSON',
    alarm_rules_json TEXT COMMENT 'Alarm Rules JSON',
    sampling_config_json TEXT COMMENT 'Sampling Config JSON',
    is_enabled BOOLEAN DEFAULT TRUE COMMENT 'Is Enabled',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated At',
    INDEX idx_chart_code (chart_code),
    INDEX idx_chart_type (chart_type),
    INDEX idx_parameter_code (parameter_code),
    INDEX idx_is_enabled (is_enabled)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = 'QC SPC Control Chart';

-- =====================
-- V10: work report
-- =====================
-- V10: Work Report (报工记录)
CREATE TABLE IF NOT EXISTS mes_work_report (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    report_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Report No',
    task_id BIGINT COMMENT 'Production Task ID',
    task_no VARCHAR(50) COMMENT 'Task No',
    work_order_id BIGINT COMMENT 'Work Order ID',
    work_order_no VARCHAR(50) COMMENT 'Work Order No',
    workstation_id BIGINT COMMENT 'Workstation ID',
    workstation_name VARCHAR(100) COMMENT 'Workstation Name',
    process_code VARCHAR(50) COMMENT 'Process Code',
    process_name VARCHAR(100) COMMENT 'Process Name',
    step_no INT COMMENT 'Step No',
    operator_id VARCHAR(50) COMMENT 'Operator ID',
    operator_name VARCHAR(100) COMMENT 'Operator Name',
    report_time DATETIME COMMENT 'Report Time',
    quantity INT COMMENT 'Total Quantity',
    qualified_quantity INT COMMENT 'Qualified Quantity',
    defective_quantity INT COMMENT 'Defective Quantity',
    duration_minutes INT COMMENT 'Duration in Minutes',
    parameter_values_json TEXT COMMENT 'Parameter Values JSON',
    remarks VARCHAR(500) COMMENT 'Remarks',
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT' COMMENT 'Status: DRAFT, SUBMITTED, QUALITY_CHECKED, CONFIRMED, CANCELLED',
    created_by VARCHAR(50) COMMENT 'Created By',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated At',
    INDEX idx_report_no (report_no),
    INDEX idx_task_id (task_id),
    INDEX idx_work_order_id (work_order_id),
    INDEX idx_workstation_id (workstation_id),
    INDEX idx_operator_id (operator_id),
    INDEX idx_status (status),
    INDEX idx_report_time (report_time)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = 'Work Report';

-- =====================
-- V11: workstation
-- =====================
-- Workstation (工位) table
CREATE TABLE IF NOT EXISTS mes_workstation (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    workstation_code VARCHAR(50) NOT NULL UNIQUE COMMENT '工位编码',
    workstation_name VARCHAR(100) NOT NULL COMMENT '工位名称',
    workshop_id VARCHAR(50) COMMENT '所属车间ID',
    workshop_name VARCHAR(100) COMMENT '所属车间名称',
    type VARCHAR(50) NOT NULL COMMENT '工位类型: ASSEMBLY/INSPECTION/PACKAGING/STORAGE/MATERIAL_PREP/TESTING/REWORK/OTHER',
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE/INACTIVE/MAINTENANCE/FAULT',
    location VARCHAR(200) COMMENT '位置描述',
    capacity INT COMMENT '产能/容量',
    description VARCHAR(500) COMMENT '描述',
    equipment_ids TEXT COMMENT '关联设备ID列表 JSON',
    equipment_codes TEXT COMMENT '关联设备编码列表 JSON',
    tool_ids TEXT COMMENT '关联工具ID列表 JSON',
    tool_names TEXT COMMENT '关联工具名称列表 JSON',
    operator_ids TEXT COMMENT '操作人员ID列表 JSON',
    operator_names TEXT COMMENT '操作人员名称列表 JSON',
    extension_fields TEXT COMMENT '扩展字段 JSON',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_workstation_code (workstation_code),
    INDEX idx_workshop_id (workshop_id),
    INDEX idx_type (type),
    INDEX idx_status (status)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '工位表';

-- =====================
-- V12: parameter_group_template & process_parameter
-- =====================
-- Parameter Group Template (工艺参数组模板) table
CREATE TABLE IF NOT EXISTS mes_parameter_group_template (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    template_code VARCHAR(50) NOT NULL UNIQUE COMMENT '模板编码',
    template_name VARCHAR(100) NOT NULL COMMENT '模板名称',
    product_type VARCHAR(50) COMMENT '关联产品类型',
    description VARCHAR(500) COMMENT '描述',
    status VARCHAR(50) NOT NULL DEFAULT 'DRAFT' COMMENT '状态: DRAFT/ACTIVE/INACTIVE/ARCHIVED',
    display_order INT DEFAULT 0 COMMENT '显示顺序',
    parameter_ids TEXT COMMENT '参数ID列表 JSON',
    parameter_codes TEXT COMMENT '参数编码列表 JSON',
    created_by VARCHAR(50) COMMENT '创建人',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_by VARCHAR(50) COMMENT '更新人',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_template_code (template_code),
    INDEX idx_product_type (product_type),
    INDEX idx_status (status)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '工艺参数组模板表';

-- Process Parameter (工艺参数) table
CREATE TABLE IF NOT EXISTS mes_process_parameter (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    param_code VARCHAR(50) NOT NULL UNIQUE COMMENT '参数编码',
    param_name VARCHAR(100) NOT NULL COMMENT '参数名称',
    route_id BIGINT COMMENT '工艺路线ID',
    route_code VARCHAR(50) COMMENT '工艺路线编码',
    step_no INT COMMENT '工序序号',
    step_code VARCHAR(50) COMMENT '工序编码',
    param_type VARCHAR(50) COMMENT '参数类型: TEMPERATURE/PRESSURE/SPEED/TIME/CURRENT/VOLTAGE/FORCE/LENGTH/ANGLE/WEIGHT/VOLUME/SPEED_PER_MINUTE/HUMIDITY/QUALITY/COUNT/OTHER',
    data_type VARCHAR(50) COMMENT '数据类型: INTEGER/DECIMAL/TEXT/BOOLEAN',
    unit VARCHAR(20) COMMENT '单位',
    standard_value DECIMAL(18,4) COMMENT '标准值',
    upper_limit DECIMAL(18,4) COMMENT '上限',
    lower_limit DECIMAL(18,4) COMMENT '下限',
    collection_method VARCHAR(50) COMMENT '采集方式: MANUAL/AUTO_SENSOR/PLC/BARCODE',
    device_address VARCHAR(100) COMMENT '设备地址',
    is_required TINYINT(1) DEFAULT 0 COMMENT '是否必填',
    validation_rule VARCHAR(200) COMMENT '校验规则',
    alarm_threshold DECIMAL(18,4) COMMENT '告警阈值(百分比)',
    display_order INT DEFAULT 0 COMMENT '显示顺序',
    param_group VARCHAR(50) COMMENT '参数分组',
    remark VARCHAR(500) COMMENT '备注',
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE/INACTIVE',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_param_code (param_code),
    INDEX idx_route_id (route_id),
    INDEX idx_route_code (route_code),
    INDEX idx_step_no (step_no),
    INDEX idx_param_type (param_type),
    INDEX idx_status (status)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '工艺参数表';

-- =====================
-- V13: process_route
-- =====================
-- Process Route (工艺路线) table
CREATE TABLE IF NOT EXISTS mes_process_route (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    route_code VARCHAR(50) NOT NULL UNIQUE COMMENT '工艺路线编码',
    route_name VARCHAR(100) NOT NULL COMMENT '工艺路线名称',
    product_code VARCHAR(50) COMMENT '产品编码',
    version INT NOT NULL DEFAULT 1 COMMENT '版本号',
    status VARCHAR(50) NOT NULL DEFAULT 'DRAFT' COMMENT '状态: DRAFT/ACTIVE/ARCHIVED',
    effective_date DATETIME COMMENT '生效日期',
    expiry_date DATETIME COMMENT '失效日期',
    steps TEXT COMMENT '工序步骤 JSON 数组',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_route_code (route_code),
    INDEX idx_product_code (product_code),
    INDEX idx_version (version),
    INDEX idx_status (status),
    INDEX idx_effective_expiry (effective_date, expiry_date)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '工艺路线表';

-- =====================
-- V14: scheduling
-- =====================
-- 排程订单表
CREATE TABLE IF NOT EXISTS mes_schedule_order (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    schedule_no VARCHAR(64) NOT NULL UNIQUE COMMENT '排程编号',
    order_no VARCHAR(64) NOT NULL COMMENT '工单编号',
    product_code VARCHAR(64) COMMENT '产品编码',
    product_name VARCHAR(128) COMMENT '产品名称',
    quantity DECIMAL(18, 4) DEFAULT 0 COMMENT '数量',
    completed_quantity DECIMAL(18, 4) DEFAULT 0 COMMENT '已完成数量',
    due_date DATETIME COMMENT '到期时间',
    scheduled_start_time DATETIME COMMENT '计划开始时间',
    scheduled_end_time DATETIME COMMENT '计划结束时间',
    actual_start_time DATETIME COMMENT '实际开始时间',
    actual_end_time DATETIME COMMENT '实际结束时间',
    priority VARCHAR(32) DEFAULT 'NORMAL' COMMENT '优先级: LOW/NORMAL/HIGH/URGENT',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT '状态: PENDING/SCHEDULED/IN_PROGRESS/COMPLETED/DELAYED/CANCELLED',
    workshop_id VARCHAR(64) COMMENT '车间ID',
    route_code VARCHAR(64) COMMENT '工艺路线编码',
    remark VARCHAR(512) COMMENT '备注',
    created_by VARCHAR(64) COMMENT '创建人',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_schedule_no (schedule_no),
    INDEX idx_order_no (order_no),
    INDEX idx_workshop_id (workshop_id),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_due_date (due_date)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '排程订单表';

-- 排程工序表
CREATE TABLE IF NOT EXISTS mes_schedule_operation (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    schedule_order_id BIGINT NOT NULL COMMENT '排程订单ID',
    operation_code VARCHAR(64) NOT NULL COMMENT '工序编码',
    operation_name VARCHAR(128) COMMENT '工序名称',
    sequence_no INT DEFAULT 1 COMMENT '工序序号',
    resource_code VARCHAR(64) COMMENT '资源编码',
    resource_name VARCHAR(128) COMMENT '资源名称',
    setup_time_minutes DECIMAL(12, 2) DEFAULT 0 COMMENT '准备时间(分钟)',
    processing_time_minutes DECIMAL(12, 2) DEFAULT 0 COMMENT '加工时间(分钟)',
    teardown_time_minutes DECIMAL(12, 2) DEFAULT 0 COMMENT '收尾时间(分钟)',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT '状态: PENDING/SCHEDULED/IN_PROGRESS/COMPLETED/BLOCKED',
    scheduled_start_time DATETIME COMMENT '计划开始时间',
    scheduled_end_time DATETIME COMMENT '计划结束时间',
    INDEX idx_schedule_order_id (schedule_order_id),
    INDEX idx_operation_code (operation_code),
    INDEX idx_resource_code (resource_code),
    INDEX idx_status (status),
    FOREIGN KEY (schedule_order_id) REFERENCES mes_schedule_order (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '排程工序表';

-- 排程资源表
CREATE TABLE IF NOT EXISTS mes_schedule_resource (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    resource_code VARCHAR(64) NOT NULL UNIQUE COMMENT '资源编码',
    resource_name VARCHAR(128) NOT NULL COMMENT '资源名称',
    resource_type VARCHAR(32) NOT NULL COMMENT '资源类型: EQUIPMENT/WORK_CENTER/LABOR/TOOL',
    status VARCHAR(32) DEFAULT 'AVAILABLE' COMMENT '状态: AVAILABLE/OCCUPIED/MAINTENANCE/OFFLINE',
    workshop_id VARCHAR(64) COMMENT '车间ID',
    capacity_per_shift DOUBLE COMMENT '每班产能',
    calendar_code VARCHAR(64) COMMENT '日历编码',
    description VARCHAR(512) COMMENT '描述',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_resource_code (resource_code),
    INDEX idx_resource_type (resource_type),
    INDEX idx_workshop_id (workshop_id),
    INDEX idx_status (status)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '排程资源表';

-- =====================
-- V15: labor & time tracking
-- =====================
-- 操作员表
CREATE TABLE IF NOT EXISTS mes_operator (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    operator_code VARCHAR(64) NOT NULL UNIQUE COMMENT '操作员编码',
    operator_name VARCHAR(128) NOT NULL COMMENT '操作员姓名',
    department VARCHAR(64) COMMENT '部门',
    job_title VARCHAR(64) COMMENT '岗位',
    shift_group VARCHAR(32) COMMENT '班次组',
    status VARCHAR(32) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE/INACTIVE/ON_LEAVE/TERMINATED',
    phone VARCHAR(32) COMMENT '电话',
    email VARCHAR(128) COMMENT '邮箱',
    id_card_no VARCHAR(32) COMMENT '身份证号',
    hire_date DATETIME COMMENT '入职日期',
    remark VARCHAR(512) COMMENT '备注',
    created_by VARCHAR(64) COMMENT '创建人',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_operator_code (operator_code),
    INDEX idx_department (department),
    INDEX idx_status (status)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '操作员表';

-- 操作员技能表
CREATE TABLE IF NOT EXISTS mes_operator_skill (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    operator_id BIGINT NOT NULL COMMENT '操作员ID',
    skill_code VARCHAR(64) NOT NULL COMMENT '技能编码',
    skill_name VARCHAR(128) COMMENT '技能名称',
    skill_level VARCHAR(32) DEFAULT 'TRAINEE' COMMENT '技能等级: TRAINEE/JUNIOR/MIDDLE/SENIOR/MASTER',
    certified BOOLEAN DEFAULT FALSE COMMENT '是否认证',
    certified_at DATETIME COMMENT '认证时间',
    expiry_at DATETIME COMMENT '到期时间',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_operator_id (operator_id),
    INDEX idx_skill_code (skill_code),
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '操作员技能表';

-- 工作中心分配表
CREATE TABLE IF NOT EXISTS mes_work_center_assignment (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    operator_id BIGINT NOT NULL COMMENT '操作员ID',
    work_center_id VARCHAR(64) NOT NULL COMMENT '工作中心ID',
    work_center_name VARCHAR(128) COMMENT '工作中心名称',
    start_date DATE COMMENT '开始日期',
    end_date DATE COMMENT '结束日期',
    shift_type VARCHAR(32) DEFAULT 'DAY' COMMENT '班次: DAY/NIGHT/MIDDLE/ROTATING',
    status VARCHAR(32) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE/INACTIVE',
    remark VARCHAR(512) COMMENT '备注',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_operator_id (operator_id),
    INDEX idx_work_center_id (work_center_id),
    INDEX idx_status (status),
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '工作中心分配表';

-- 工时记录表
CREATE TABLE IF NOT EXISTS mes_time_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    operator_id BIGINT NOT NULL COMMENT '操作员ID',
    operator_code VARCHAR(64) COMMENT '操作员编码',
    operator_name VARCHAR(128) COMMENT '操作员姓名',
    work_order_no VARCHAR(64) COMMENT '工单编号',
    task_no VARCHAR(64) COMMENT '任务编号',
    operation_code VARCHAR(64) COMMENT '工序编码',
    work_center_id VARCHAR(64) COMMENT '工作中心ID',
    record_date DATE COMMENT '记录日期',
    start_time DATETIME COMMENT '开始时间',
    end_time DATETIME COMMENT '结束时间',
    total_hours DECIMAL(12, 2) DEFAULT 0 COMMENT '总工时(小时)',
    record_type VARCHAR(32) DEFAULT 'REGULAR' COMMENT '类型: REGULAR/OVERTIME/VACATION/SICK',
    status VARCHAR(32) DEFAULT 'DRAFT' COMMENT '状态: DRAFT/SUBMITTED/APPROVED/REJECTED',
    approved_by VARCHAR(64) COMMENT '审批人',
    approved_at DATETIME COMMENT '审批时间',
    remark VARCHAR(512) COMMENT '备注',
    created_by VARCHAR(64) COMMENT '创建人',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_operator_id (operator_id),
    INDEX idx_record_date (record_date),
    INDEX idx_work_order_no (work_order_no),
    INDEX idx_status (status),
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '工时记录表';

-- 考勤表
CREATE TABLE IF NOT EXISTS mes_attendance (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    operator_id BIGINT NOT NULL COMMENT '操作员ID',
    operator_code VARCHAR(64) COMMENT '操作员编码',
    operator_name VARCHAR(128) COMMENT '操作员姓名',
    attendance_date DATE COMMENT '考勤日期',
    clock_in TIME COMMENT '签到时间',
    clock_out TIME COMMENT '签退时间',
    scheduled_start TIME COMMENT '应到时间',
    scheduled_end TIME COMMENT '应退时间',
    status VARCHAR(32) DEFAULT 'ABSENT' COMMENT '状态: PRESENT/LATE/ABSENT/HALF_DAY/OVERTIME/VACATION/SICK/BUSINESS_TRIP',
    overtime BOOLEAN DEFAULT FALSE COMMENT '是否加班',
    remark VARCHAR(512) COMMENT '备注',
    created_by VARCHAR(64) COMMENT '创建人',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_operator_id (operator_id),
    INDEX idx_attendance_date (attendance_date),
    INDEX idx_status (status),
    FOREIGN KEY (operator_id) REFERENCES mes_operator (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '考勤表';