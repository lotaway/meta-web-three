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

CREATE INDEX idx_dashboard_template_status ON mes_dashboard_template(status);
CREATE INDEX idx_dashboard_template_type ON mes_dashboard_template(template_type);
CREATE INDEX idx_dashboard_component_type ON mes_dashboard_component(component_type);