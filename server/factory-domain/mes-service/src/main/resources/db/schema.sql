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
    status_code VARCHAR(50) COMMENT '可配置状态机的状态码',
    type_code VARCHAR(50) DEFAULT 'NORMAL' COMMENT '工单类型: NORMAL, REWORK, REPAIR, SAMPLE',
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
CREATE INDEX idx_work_order_type_code ON mes_work_order(type_code);
CREATE INDEX idx_work_order_status_code ON mes_work_order(status_code);
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

-- 工单类型表
CREATE TABLE IF NOT EXISTS mes_work_order_type (
    id BIGSERIAL PRIMARY KEY,
    type_code VARCHAR(50) NOT NULL UNIQUE COMMENT '类型编码: NORMAL, REWORK, REPAIR, SAMPLE',
    type_name VARCHAR(100) NOT NULL COMMENT '类型名称',
    description VARCHAR(500) COMMENT '描述',
    status_machine_code VARCHAR(50) COMMENT '关联的状态机编码',
    process_route_template VARCHAR(50) COMMENT '工艺路线模板编码',
    is_default BOOLEAN DEFAULT FALSE COMMENT '是否为默认类型',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 状态机配置表
CREATE TABLE IF NOT EXISTS mes_status_machine (
    id BIGSERIAL PRIMARY KEY,
    machine_code VARCHAR(50) NOT NULL UNIQUE COMMENT '状态机编码',
    machine_name VARCHAR(100) NOT NULL COMMENT '状态机名称',
    entity_type VARCHAR(50) NOT NULL COMMENT '实体类型: WORK_ORDER, PRODUCTION_TASK, EQUIPMENT',
    description VARCHAR(500) COMMENT '描述',
    initial_status VARCHAR(50) COMMENT '初始状态',
    is_default BOOLEAN DEFAULT FALSE COMMENT '是否为默认状态机',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 状态配置表
CREATE TABLE IF NOT EXISTS mes_status_config (
    id BIGSERIAL PRIMARY KEY,
    machine_id BIGINT NOT NULL REFERENCES mes_status_machine(id) ON DELETE CASCADE,
    status_code VARCHAR(50) NOT NULL COMMENT '状态编码',
    status_name VARCHAR(100) NOT NULL COMMENT '状态名称',
    status_category VARCHAR(20) COMMENT '状态分类: PENDING, PROCESSING, COMPLETED, CANCELLED, SPECIAL',
    is_initial BOOLEAN DEFAULT FALSE COMMENT '是否为初始状态',
    is_final BOOLEAN DEFAULT FALSE COMMENT '是否为终态',
    color VARCHAR(20) COMMENT '前端显示颜色',
    icon VARCHAR(50) COMMENT '前端显示图标',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(machine_id, status_code)
);

-- 状态转换规则表
CREATE TABLE IF NOT EXISTS mes_status_transition (
    id BIGSERIAL PRIMARY KEY,
    machine_id BIGINT NOT NULL REFERENCES mes_status_machine(id) ON DELETE CASCADE,
    from_status VARCHAR(50) NOT NULL COMMENT '源状态',
    to_status VARCHAR(50) NOT NULL COMMENT '目标状态',
    transition_action VARCHAR(50) NOT NULL COMMENT '转换动作: RELEASE, START, PAUSE, RESUME, COMPLETE, CANCEL',
    condition_expression VARCHAR(500) COMMENT '触发条件表达式',
    event_code VARCHAR(50) COMMENT '触发事件编码',
    is_auto_transition BOOLEAN DEFAULT FALSE COMMENT '是否自动转换',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(machine_id, from_status, to_status)
);

-- 创建工单类型表索引
CREATE INDEX idx_work_order_type_code ON mes_work_order_type(type_code);
CREATE INDEX idx_work_order_type_status ON mes_work_order_type(status);

-- 创建状态机配置表索引
CREATE INDEX idx_status_machine_code ON mes_status_machine(machine_code);
CREATE INDEX idx_status_machine_entity_type ON mes_status_machine(entity_type);

-- 创建状态配置表索引
CREATE INDEX idx_status_config_machine_id ON mes_status_config(machine_id);

-- 创建状态转换规则表索引
CREATE INDEX idx_status_transition_machine_id ON mes_status_transition(machine_id);
-- 产品SN规则绑定表
CREATE TABLE IF NOT EXISTS mes_product_sn_rule (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL COMMENT '产品ID',
    product_code VARCHAR(50) NOT NULL COMMENT '产品编码',
    code_rule_id BIGINT NOT NULL COMMENT '编码规则ID',
    rule_code VARCHAR(50) NOT NULL COMMENT '编码规则代码',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    description VARCHAR(500) COMMENT '描述',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id)
);

-- 创建产品SN规则绑定表索引
CREATE INDEX idx_product_sn_rule_product_id ON mes_product_sn_rule(product_id);
CREATE INDEX idx_product_sn_rule_product_code ON mes_product_sn_rule(product_code);
CREATE INDEX idx_product_sn_rule_code_rule_id ON mes_product_sn_rule(code_rule_id);
CREATE INDEX idx_product_sn_rule_active ON mes_product_sn_rule(is_active);

-- 工单编码规则绑定表
CREATE TABLE IF NOT EXISTS mes_work_order_code_rule (
    id BIGSERIAL PRIMARY KEY,
    workshop_id VARCHAR(50) COMMENT '车间ID',
    work_order_type VARCHAR(50) NOT NULL COMMENT '工单类型: NORMAL, REWORK, REPAIR, SAMPLE',
    code_rule_id BIGINT NOT NULL COMMENT '编码规则ID',
    rule_code VARCHAR(50) NOT NULL COMMENT '编码规则代码',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    description VARCHAR(500) COMMENT '描述',
    priority INTEGER DEFAULT 0 COMMENT '优先级，数字越大优先级越高',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建工单编码规则绑定表索引
CREATE INDEX idx_work_order_code_rule_workshop_type ON mes_work_order_code_rule(workshop_id, work_order_type);
CREATE INDEX idx_work_order_code_rule_type ON mes_work_order_code_rule(work_order_type);
CREATE INDEX idx_work_order_code_rule_workshop ON mes_work_order_code_rule(workshop_id);
CREATE INDEX idx_work_order_code_rule_code_rule_id ON mes_work_order_code_rule(code_rule_id);
CREATE INDEX idx_work_order_code_rule_active ON mes_work_order_code_rule(is_active);

-- 为工单表添加编码规则绑定字段
ALTER TABLE mes_work_order ADD COLUMN IF NOT EXISTS code_rule_id BIGINT;

-- 领料模式配置表
CREATE TABLE IF NOT EXISTS mes_material_issue_config (
    id BIGSERIAL PRIMARY KEY,
    config_code VARCHAR(50) NOT NULL UNIQUE COMMENT '配置编码',
    config_name VARCHAR(100) NOT NULL COMMENT '配置名称',
    workshop_id VARCHAR(50) COMMENT '车间ID',
    product_code VARCHAR(50) COMMENT '产品编码，为空表示车间级默认配置',
    issue_mode VARCHAR(20) NOT NULL COMMENT '领料模式: PRE_PICKING(备料制), PULL(领料制), JIT(JIT配送)',
    issue_rule VARCHAR(20) DEFAULT 'FIFO' COMMENT '发料规则: FIFO(先进先出), LIFO(后进先出), LOCKED_BATCH(锁定批号), EXPIRY_FIRST(效期优先)',
    lead_time_hours INTEGER DEFAULT 0 COMMENT '提前期(小时)，备料制和JIT模式使用',
    buffer_hours INTEGER DEFAULT 0 COMMENT '缓冲期(小时)，JIT模式使用',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    priority INTEGER DEFAULT 0 COMMENT '优先级，数字越大优先级越高，用于匹配优先级',
    description VARCHAR(500) COMMENT '描述',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workshop_id, product_code)
);

-- 创建领料模式配置表索引
CREATE INDEX idx_material_issue_config_workshop_product ON mes_material_issue_config(workshop_id, product_code);
CREATE INDEX idx_material_issue_config_workshop ON mes_material_issue_config(workshop_id);
CREATE INDEX idx_material_issue_config_product_code ON mes_material_issue_config(product_code);
CREATE INDEX idx_material_issue_config_issue_mode ON mes_material_issue_config(issue_mode);
CREATE INDEX idx_material_issue_config_active ON mes_material_issue_config(is_active);
CREATE INDEX idx_material_issue_config_priority ON mes_material_issue_config(priority DESC);

-- 质检方案表
CREATE TABLE IF NOT EXISTS mes_qc_inspection_plan (
    id BIGSERIAL PRIMARY KEY,
    plan_code VARCHAR(50) NOT NULL UNIQUE COMMENT '方案编码',
    plan_name VARCHAR(100) NOT NULL COMMENT '方案名称',
    inspection_type VARCHAR(20) NOT NULL COMMENT '检验类型: IQC, IPQC, FQC, OQC',
    applicable_product_types VARCHAR(500) COMMENT '适用产品类型，多个用逗号分隔',
    version INTEGER DEFAULT 1 COMMENT '版本号',
    sampling_plan_code VARCHAR(50) COMMENT '抽样方案编码',
    sampling_type VARCHAR(30) DEFAULT 'RANDOM_SAMPLING' COMMENT '抽样类型: FULL_INSPECTION, RANDOM_SAMPLING, SYSTEMATIC_SAMPLING, DOUBLE_SAMPLING',
    aql VARCHAR(20) DEFAULT '0.65' COMMENT 'AQL值',
    inspection_level VARCHAR(20) DEFAULT 'normal' COMMENT '检验水平: normal, reduced, tightened',
    sample_size INTEGER DEFAULT 0 COMMENT '样本大小，0表示根据AQL计算',
    accept_number VARCHAR(20) DEFAULT '0' COMMENT '合格判定数',
    reject_number VARCHAR(20) DEFAULT '1' COMMENT '不合格判定数',
    disposition_rule VARCHAR(500) COMMENT '处置规则',
    qualified_flow VARCHAR(50) DEFAULT 'pass' COMMENT '合格流向',
    unqualified_flow VARCHAR(50) DEFAULT 'isolate' COMMENT '不合格流向',
    special_approval_flow VARCHAR(500) COMMENT '特采审批流程',
    status VARCHAR(20) DEFAULT 'DRAFT' COMMENT '状态: DRAFT, EFFECTIVE, EXPIRED, CANCELLED',
    effective_date TIMESTAMP COMMENT '生效日期',
    expiration_date TIMESTAMP COMMENT '失效日期',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    remark VARCHAR(500) COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_inspection_plan_code ON mes_qc_inspection_plan(plan_code);
CREATE INDEX idx_inspection_plan_type ON mes_qc_inspection_plan(inspection_type);
CREATE INDEX idx_inspection_plan_status ON mes_qc_inspection_plan(status);
CREATE INDEX idx_inspection_plan_product_types ON mes_qc_inspection_plan(applicable_product_types);

-- 检验项表（检验项库）
CREATE TABLE IF NOT EXISTS mes_qc_inspection_item (
    id BIGSERIAL PRIMARY KEY,
    item_code VARCHAR(50) NOT NULL UNIQUE COMMENT '检验项编码',
    item_name VARCHAR(100) NOT NULL COMMENT '检验项名称',
    item_category VARCHAR(50) COMMENT '检验项分类: appearance, dimension, function, performance, safety',
    data_type VARCHAR(20) DEFAULT 'NUMERIC' COMMENT '数据类型: NUMERIC, TEXT, BOOLEAN, DATE, SELECT',
    unit VARCHAR(20) COMMENT '单位',
    standard_value DECIMAL(20,4) COMMENT '标准值',
    upper_limit DECIMAL(20,4) COMMENT '规格上限',
    lower_limit DECIMAL(20,4) COMMENT '规格下限',
    inspection_method VARCHAR(200) COMMENT '检验方法',
    inspection_tool VARCHAR(100) COMMENT '检测工具',
    severity INTEGER DEFAULT 1 COMMENT '严重等级: 1-轻微, 2-一般, 3-严重, 4-致命',
    is_mandatory BOOLEAN DEFAULT TRUE COMMENT '是否必检',
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT '状态: ACTIVE, INACTIVE',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    remark VARCHAR(500) COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_inspection_item_code ON mes_qc_inspection_item(item_code);
CREATE INDEX idx_inspection_item_category ON mes_qc_inspection_item(item_category);
CREATE INDEX idx_inspection_item_status ON mes_qc_inspection_item(status);

-- 方案检验项关联表
CREATE TABLE IF NOT EXISTS mes_qc_plan_item (
    id BIGSERIAL PRIMARY KEY,
    plan_id BIGINT NOT NULL COMMENT '质检方案ID',
    item_id BIGINT NOT NULL COMMENT '检验项ID',
    item_sequence INTEGER NOT NULL COMMENT '检验项序号',
    is_mandatory BOOLEAN DEFAULT TRUE COMMENT '是否必检',
    default_value VARCHAR(200) COMMENT '默认值',
    inspection_method VARCHAR(200) COMMENT '检验方法，覆盖检验项默认方法',
    sampling_rule VARCHAR(500) COMMENT '抽样规则',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(plan_id, item_id)
);

CREATE INDEX idx_plan_item_plan ON mes_qc_plan_item(plan_id);
CREATE INDEX idx_plan_item_item ON mes_qc_plan_item(item_id);

-- SOP文档表
CREATE TABLE IF NOT EXISTS mes_sop_document (
    id BIGSERIAL PRIMARY KEY,
    document_code VARCHAR(50) NOT NULL UNIQUE COMMENT '文档编码',
    document_name VARCHAR(100) NOT NULL COMMENT '文档名称',
    document_type VARCHAR(50) COMMENT '文档类型: OPERATING_INSTRUCTION, MAINTENANCE_MANUAL, SAFETY_GUIDE, TRAINING_MATERIAL',
    category VARCHAR(50) COMMENT '分类',
    current_version INTEGER DEFAULT 0 COMMENT '当前版本号',
    status VARCHAR(20) DEFAULT 'DRAFT' COMMENT '状态: DRAFT, ACTIVE, ARCHIVED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sop_document_code ON mes_sop_document(document_code);
CREATE INDEX idx_sop_document_type ON mes_sop_document(document_type);
CREATE INDEX idx_sop_document_category ON mes_sop_document(category);
CREATE INDEX idx_sop_document_status ON mes_sop_document(status);

-- SOP文档版本表
CREATE TABLE IF NOT EXISTS mes_sop_document_version (
    id BIGSERIAL PRIMARY KEY,
    sop_document_id BIGINT NOT NULL REFERENCES mes_sop_document(id) ON DELETE CASCADE,
    version_no INTEGER NOT NULL COMMENT '版本号',
    file_name VARCHAR(200) NOT NULL COMMENT '文件名',
    file_path VARCHAR(500) NOT NULL COMMENT '文件路径',
    file_type VARCHAR(20) COMMENT '文件类型: PDF, DOC, DOCX, PPT, PPTX, VIDEO, IMG',
    file_size BIGINT COMMENT '文件大小(字节)',
    uploader VARCHAR(50) COMMENT '上传人',
    change_description VARCHAR(500) COMMENT '变更说明',
    is_current_version BOOLEAN DEFAULT FALSE COMMENT '是否为当前版本',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sop_document_id, version_no)
);

CREATE INDEX idx_sop_document_version_document ON mes_sop_document_version(sop_document_id);
CREATE INDEX idx_sop_document_version_current ON mes_sop_document_version(sop_document_id, is_current_version);

-- SOP工艺路线绑定表（工序/工位关联）
CREATE TABLE IF NOT EXISTS mes_sop_route_binding (
    id BIGSERIAL PRIMARY KEY,
    sop_document_id BIGINT NOT NULL REFERENCES mes_sop_document(id) ON DELETE CASCADE,
    route_code VARCHAR(50) COMMENT '工艺路线编码',
    route_name VARCHAR(100) COMMENT '工艺路线名称',
    step_no INTEGER COMMENT '工序号',
    process_code VARCHAR(50) COMMENT '工序编码',
    process_name VARCHAR(100) COMMENT '工序名称',
    workstation_id VARCHAR(50) COMMENT '工位ID',
    workstation_name VARCHAR(100) COMMENT '工位名称',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sop_route_binding_document ON mes_sop_route_binding(sop_document_id);
CREATE INDEX idx_sop_route_binding_route ON mes_sop_route_binding(route_code, step_no);
CREATE INDEX idx_sop_route_binding_workstation ON mes_sop_route_binding(workstation_id);
CREATE INDEX idx_sop_route_binding_active ON mes_sop_route_binding(is_active);

-- 设备保养计划表
CREATE TABLE IF NOT EXISTS mes_equipment_maintenance_plan (
    id BIGSERIAL PRIMARY KEY,
    plan_code VARCHAR(50) NOT NULL UNIQUE COMMENT '计划编码',
    plan_name VARCHAR(100) NOT NULL COMMENT '计划名称',
    description VARCHAR(500) COMMENT '描述',
    equipment_type_id BIGINT COMMENT '设备类型ID',
    equipment_type_code VARCHAR(50) COMMENT '设备类型编码',
    cycle_type VARCHAR(20) NOT NULL COMMENT '周期类型: TIME_BASED, RUNNING_HOURS',
    cycle_days INTEGER COMMENT '周期天数（时间周期类型使用）',
    cycle_running_hours INTEGER COMMENT '周期运行时长（运行时长类型使用）',
    advance_alert_days INTEGER DEFAULT 3 COMMENT '提前预警天数',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_maintenance_plan_code ON mes_equipment_maintenance_plan(plan_code);
CREATE INDEX idx_maintenance_plan_type ON mes_equipment_maintenance_plan(equipment_type_code);
CREATE INDEX idx_maintenance_plan_active ON mes_equipment_maintenance_plan(is_active);

-- 设备保养项目明细表
CREATE TABLE IF NOT EXISTS mes_equipment_maintenance_item (
    id BIGSERIAL PRIMARY KEY,
    plan_id BIGINT NOT NULL REFERENCES mes_equipment_maintenance_plan(id) ON DELETE CASCADE,
    item_code VARCHAR(50) NOT NULL COMMENT '项目编码',
    item_name VARCHAR(100) NOT NULL COMMENT '项目名称',
    description VARCHAR(500) COMMENT '项目描述',
    check_method VARCHAR(200) COMMENT '检查方法',
    standard VARCHAR(500) COMMENT '标准/判定依据',
    is_required BOOLEAN DEFAULT TRUE COMMENT '是否必检',
    sort_order INTEGER DEFAULT 0 COMMENT '排序序号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_maintenance_item_plan ON mes_equipment_maintenance_item(plan_id);
CREATE INDEX idx_maintenance_item_code ON mes_equipment_maintenance_item(item_code);
