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

CREATE INDEX idx_code_rule_element_rule_id ON mes_code_rule_element(rule_id);

-- 创建索引
CREATE INDEX idx_extension_field_entity_type ON mes_entity_extension_field(entity_type);
CREATE INDEX idx_extension_field_value_entity ON mes_entity_extension_field_value(entity_type, entity_id);
CREATE INDEX idx_data_dictionary_item_dict_id ON mes_data_dictionary_item(dict_id);
CREATE INDEX idx_code_rule_business_type ON mes_code_rule(business_type);