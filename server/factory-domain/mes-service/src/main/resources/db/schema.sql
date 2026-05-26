-- 扩展字段定义表
CREATE TABLE IF NOT EXISTS mes_entity_extension_field (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL COMMENT '实体类型',
    field_code VARCHAR(50) NOT NULL COMMENT '字段编码',
    field_name VARCHAR(100) NOT NULL COMMENT '字段名称',
    field_type VARCHAR(20) NOT NULL COMMENT '字段类型: TEXT, NUMBER, DATE, DATETIME, SELECT, MULTI_SELECT, CHECKBOX, SWITCH, REFERENCE',
    default_value VARCHAR(500) COMMENT '默认值',
    required BOOLEAN DEFAULT FALSE COMMENT '是否必填',
    is_unique BOOLEAN DEFAULT FALSE COMMENT '是否唯一',
    validation_rule VARCHAR(500) COMMENT '校验规则(正则表达式)',
    list_visible BOOLEAN DEFAULT TRUE COMMENT '是否列表显示',
    searchable BOOLEAN DEFAULT FALSE COMMENT '是否可搜索',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    field_group VARCHAR(50) COMMENT '字段分组',
    reference_type VARCHAR(50) COMMENT '引用类型(REFERENCE时)',
    reference_entity VARCHAR(50) COMMENT '引用实体(REFERENCE时)',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, field_code)
);

-- 扩展字段值表
CREATE TABLE IF NOT EXISTS mes_entity_extension_field_value (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL COMMENT '实体类型',
    entity_id BIGINT NOT NULL COMMENT '实体ID',
    field_code VARCHAR(50) NOT NULL COMMENT '字段编码',
    field_value TEXT COMMENT '字段值',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_id, field_code)
);

-- 数据字典表
CREATE TABLE IF NOT EXISTS mes_data_dictionary (
    id BIGSERIAL PRIMARY KEY,
    dict_code VARCHAR(50) NOT NULL UNIQUE COMMENT '字典编码',
    dict_name VARCHAR(100) NOT NULL COMMENT '字典名称',
    description VARCHAR(500) COMMENT '字典描述',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 数据字典项表
CREATE TABLE IF NOT EXISTS mes_data_dictionary_item (
    id BIGSERIAL PRIMARY KEY,
    dict_id BIGINT NOT NULL REFERENCES mes_data_dictionary(id) ON DELETE CASCADE,
    item_code VARCHAR(50) NOT NULL COMMENT '选项编码',
    item_label VARCHAR(100) NOT NULL COMMENT '选项标签',
    parent_item_code VARCHAR(50) COMMENT '父级选项编码(级联选择)',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dict_id, item_code)
);

-- 编码规则表
CREATE TABLE IF NOT EXISTS mes_code_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_code VARCHAR(50) NOT NULL UNIQUE COMMENT '规则编码',
    rule_name VARCHAR(100) NOT NULL COMMENT '规则名称',
    business_type VARCHAR(50) NOT NULL COMMENT '适用业务类型',
    rule_expression VARCHAR(200) NOT NULL COMMENT '规则表达式',
    start_value BIGINT DEFAULT 1 COMMENT '起始值',
    current_value BIGINT DEFAULT 1 COMMENT '当前值',
    step INTEGER DEFAULT 1 COMMENT '步长',
    padding_length INTEGER DEFAULT 4 COMMENT '填充长度',
    elements TEXT COMMENT '规则要素(JSON格式存储)',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 编码规则要素表
CREATE TABLE IF NOT EXISTS mes_code_rule_element (
    id BIGSERIAL PRIMARY KEY,
    rule_id BIGINT NOT NULL REFERENCES mes_code_rule(id) ON DELETE CASCADE,
    element_type VARCHAR(20) NOT NULL COMMENT '要素类型: PREFIX, DATE, SEQUENCE, BUSINESS_FIELD, DELIMITER',
    element_value VARCHAR(100) COMMENT '要素值',
    field_name VARCHAR(50) COMMENT '业务字段名(BUSINESS_FIELD时)',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号'
);

-- 工艺参数表
CREATE TABLE IF NOT EXISTS mes_process_parameter (
    id BIGSERIAL PRIMARY KEY,
    param_code VARCHAR(50) NOT NULL COMMENT '参数编码',
    param_name VARCHAR(100) NOT NULL COMMENT '参数名称',
    route_id BIGINT COMMENT '工艺路线ID',
    route_code VARCHAR(50) COMMENT '工艺路线编码',
    step_no INTEGER COMMENT '工序号',
    step_code VARCHAR(50) COMMENT '工序编码',
    param_type VARCHAR(30) COMMENT '参数类型: TEMPERATURE, PRESSURE, SPEED, TIME, CURRENT, VOLTAGE, FORCE, LENGTH, ANGLE, WEIGHT, VOLUME, SPEED_PER_MINUTE, HUMIDITY, QUALITY, COUNT, OTHER',
    data_type VARCHAR(20) COMMENT '数据类型: INTEGER, DECIMAL, TEXT, BOOLEAN',
    unit VARCHAR(20) COMMENT '单位',
    standard_value DECIMAL(20,4) COMMENT '标准值',
    upper_limit DECIMAL(20,4) COMMENT '上限',
    lower_limit DECIMAL(20,4) COMMENT '下限',
    collection_method VARCHAR(20) COMMENT '采集方式: MANUAL, AUTO_SENSOR, PLC, BARCODE',
    device_address VARCHAR(100) COMMENT '设备地址',
    is_required BOOLEAN DEFAULT FALSE COMMENT '是否必填',
    validation_rule VARCHAR(500) COMMENT '校验规则',
    alarm_threshold DECIMAL(10,2) COMMENT '报警阈值(百分比)',
    display_order INTEGER DEFAULT 0 COMMENT '显示顺序',
    param_group VARCHAR(50) COMMENT '参数分组',
    remark VARCHAR(500) COMMENT '备注',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(param_code)
);

-- 创建索引
CREATE INDEX idx_process_parameter_route_id ON mes_process_parameter(route_id);
CREATE INDEX idx_process_parameter_route_code ON mes_process_parameter(route_code);
CREATE INDEX idx_process_parameter_step_no ON mes_process_parameter(step_no);
CREATE INDEX idx_process_parameter_param_type ON mes_process_parameter(param_type);
CREATE INDEX idx_process_parameter_status ON mes_process_parameter(status);

-- 创建索引
CREATE INDEX idx_extension_field_entity_type ON mes_entity_extension_field(entity_type);
CREATE INDEX idx_extension_field_value_entity ON mes_entity_extension_field_value(entity_type, entity_id);
CREATE INDEX idx_data_dictionary_item_dict_id ON mes_data_dictionary_item(dict_id);
CREATE INDEX idx_code_rule_business_type ON mes_code_rule(business_type);

-- 工单表
CREATE TABLE IF NOT EXISTS mes_work_order (
    id BIGSERIAL PRIMARY KEY,
    work_order_no VARCHAR(50) NOT NULL UNIQUE COMMENT '工单号',
    product_code VARCHAR(50) NOT NULL COMMENT '产品编码',
    product_name VARCHAR(100) NOT NULL COMMENT '产品名称',
    quantity INTEGER NOT NULL COMMENT '计划数量',
    completed_quantity INTEGER DEFAULT 0 COMMENT '完成数量',
    status VARCHAR(20) DEFAULT 'DRAFT' COMMENT '状态: DRAFT, RELEASED, IN_PROGRESS, PAUSED, COMPLETED, CANCELLED',
    priority VARCHAR(20) DEFAULT 'NORMAL' COMMENT '优先级: LOW, NORMAL, HIGH, URGENT',
    workshop_id VARCHAR(50) COMMENT '车间ID',
    process_route_id VARCHAR(50) COMMENT '工艺路线ID',
    planned_start_time TIMESTAMP COMMENT '计划开始时间',
    planned_end_time TIMESTAMP COMMENT '计划结束时间',
    actual_start_time TIMESTAMP COMMENT '实际开始时间',
    actual_end_time TIMESTAMP COMMENT '实际结束时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 工单表索引
CREATE INDEX idx_work_order_status ON mes_work_order(status);
CREATE INDEX idx_work_order_workshop_id ON mes_work_order(workshop_id);
CREATE INDEX idx_work_order_product_code ON mes_work_order(product_code);

-- 生产任务表
CREATE TABLE IF NOT EXISTS mes_production_task (
    id BIGSERIAL PRIMARY KEY,
    task_no VARCHAR(50) NOT NULL UNIQUE COMMENT '任务号',
    work_order_id BIGINT NOT NULL COMMENT '工单ID',
    work_order_no VARCHAR(50) NOT NULL COMMENT '工单号',
    workstation_id VARCHAR(50) COMMENT '工位ID',
    workstation_name VARCHAR(100) COMMENT '工位名称',
    step_no INTEGER COMMENT '工序号',
    step_code VARCHAR(50) COMMENT '工序编码',
    step_name VARCHAR(100) COMMENT '工序名称',
    assigned_to VARCHAR(50) COMMENT '分配给(人员)',
    status VARCHAR(20) DEFAULT 'PENDING' COMMENT '状态: PENDING, ASSIGNED, STARTED, COMPLETED, CANCELLED, PAUSED',
    planned_quantity INTEGER DEFAULT 0 COMMENT '计划数量',
    completed_quantity INTEGER DEFAULT 0 COMMENT '完成数量',
    qualified_quantity INTEGER DEFAULT 0 COMMENT '合格数量',
    rejected_quantity INTEGER DEFAULT 0 COMMENT '不合格数量',
    start_time TIMESTAMP COMMENT '开始时间',
    end_time TIMESTAMP COMMENT '结束时间',
    actual_duration_minutes INTEGER COMMENT '实际耗时(分钟)',
    remarks TEXT COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 生产任务表索引
CREATE INDEX idx_production_task_work_order_id ON mes_production_task(work_order_id);
CREATE INDEX idx_production_task_work_order_no ON mes_production_task(work_order_no);
CREATE INDEX idx_production_task_workstation_id ON mes_production_task(workstation_id);
CREATE INDEX idx_production_task_status ON mes_production_task(status);
CREATE INDEX idx_production_task_task_no ON mes_production_task(task_no);

-- 设备表
CREATE TABLE IF NOT EXISTS mes_equipment (
    id BIGSERIAL PRIMARY KEY,
    equipment_code VARCHAR(50) NOT NULL UNIQUE COMMENT '设备编码',
    equipment_name VARCHAR(100) NOT NULL COMMENT '设备名称',
    equipment_type VARCHAR(50) COMMENT '设备类型',
    workshop_id VARCHAR(50) COMMENT '车间ID',
    workstation_id VARCHAR(50) COMMENT '工位ID',
    status VARCHAR(20) DEFAULT 'IDLE' COMMENT '状态: IDLE, RUNNING, MAINTENANCE, BREAKDOWN, SCRAP',
    utilization_rate DECIMAL(5,2) DEFAULT 0 COMMENT '利用率(百分比)',
    today_output INTEGER DEFAULT 0 COMMENT '今日产出',
    current_task_no VARCHAR(50) COMMENT '当前任务号',
    last_maintenance_time TIMESTAMP COMMENT '上次保养时间',
    next_maintenance_time TIMESTAMP COMMENT '下次保养时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 工艺路线表
CREATE TABLE IF NOT EXISTS mes_process_route (
    id BIGSERIAL PRIMARY KEY,
    route_code VARCHAR(50) NOT NULL UNIQUE COMMENT '工艺路线编码',
    route_name VARCHAR(100) NOT NULL COMMENT '工艺路线名称',
    product_code VARCHAR(50) NOT NULL COMMENT '产品编码',
    version INTEGER DEFAULT 1 COMMENT '版本号',
    status VARCHAR(20) DEFAULT 'DRAFT' COMMENT '状态: DRAFT, ACTIVE, ARCHIVED',
    steps JSONB COMMENT '工序步骤(JSON数组)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 工艺路线表索引
CREATE INDEX idx_process_route_code ON mes_process_route(route_code);
CREATE INDEX idx_process_route_product_code ON mes_process_route(product_code);
CREATE INDEX idx_process_route_status ON mes_process_route(status);

-- 设备表索引
CREATE INDEX idx_equipment_code ON mes_equipment(equipment_code);
CREATE INDEX idx_equipment_workshop_id ON mes_equipment(workshop_id);
CREATE INDEX idx_equipment_workstation_id ON mes_equipment(workstation_id);
CREATE INDEX idx_equipment_status ON mes_equipment(status);