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

CREATE INDEX idx_report_template_status ON mes_report_template(status);
CREATE INDEX idx_report_template_type ON mes_report_template(report_type);
CREATE INDEX idx_report_datasource_type ON mes_report_datasource(datasource_type);