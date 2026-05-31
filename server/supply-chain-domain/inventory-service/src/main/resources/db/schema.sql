-- inventory-service 数据库初始化脚本

-- 库存主表
CREATE TABLE IF NOT EXISTS inventory (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    total_quantity INT DEFAULT 0 COMMENT '总数量',
    available_quantity INT DEFAULT 0 COMMENT '可用数量',
    reserved_quantity INT DEFAULT 0 COMMENT '预留数量',
    defective_quantity INT DEFAULT 0 COMMENT '不良品数量',
    unit_cost DECIMAL(10,2) DEFAULT 0 COMMENT '单位成本',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    version INT DEFAULT 0 COMMENT '乐观锁版本',
    UNIQUE KEY uk_sku_warehouse (sku_code, warehouse_id),
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id)
) COMMENT '库存主表';

-- 库存流水表
CREATE TABLE IF NOT EXISTS inventory_record (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    biz_type VARCHAR(32) NOT NULL COMMENT '业务类型：RESERVE/CONFIRM/CANCEL/INCREASE/DECREASE',
    biz_id VARCHAR(64) COMMENT '业务单据ID',
    quantity INT NOT NULL COMMENT '变动数量',
    before_quantity INT COMMENT '变动前数量',
    after_quantity INT COMMENT '变动后数量',
    remark VARCHAR(512) COMMENT '备注',
    operator VARCHAR(64) COMMENT '操作人',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_biz_type (biz_type),
    INDEX idx_biz_id (biz_id),
    INDEX idx_created_at (created_at)
) COMMENT '库存流水表';

-- 补货建议表
CREATE TABLE IF NOT EXISTS replenishment_recommendation (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    current_stock INT COMMENT '当前库存',
    safety_stock INT COMMENT '安全库存',
    lead_time_days INT COMMENT '采购提前期(天)',
    average_daily_sales INT COMMENT '日均销量',
    recommended_quantity INT COMMENT '建议补货数量',
    recommendation_type VARCHAR(32) DEFAULT 'AUTO' COMMENT '建议类型：AUTO/MANUAL',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT '状态：PENDING/APPROVED/REJECTED',
    generated_at TIMESTAMP COMMENT '生成时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_status (status),
    INDEX idx_generated_at (generated_at)
) COMMENT '补货建议表';

-- 需求预测表
CREATE TABLE IF NOT EXISTS demand_forecast (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    forecast_period_days INT COMMENT '预测周期(天)',
    predicted_quantity INT COMMENT '预测数量',
    confidence_level INT COMMENT '置信度(0-100)',
    forecast_method VARCHAR(32) DEFAULT 'SMA' COMMENT '预测方法：SMA/WMA/EXPONENTIAL_SMOOTHING',
    forecast_start_date DATE COMMENT '预测开始日期',
    forecast_end_date DATE COMMENT '预测结束日期',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT '状态：PENDING/APPROVED/REJECTED',
    generated_at TIMESTAMP COMMENT '生成时间',
    notes VARCHAR(512) COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_status (status),
    INDEX idx_forecast_dates (forecast_start_date, forecast_end_date)
) COMMENT '需求预测表';

-- 库存批次表
CREATE TABLE IF NOT EXISTS inventory_batch (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    batch_no VARCHAR(64) NOT NULL COMMENT '批次号',
    quantity INT NOT NULL COMMENT '批次总数量',
    available_quantity INT DEFAULT 0 COMMENT '可用数量',
    reserved_quantity INT DEFAULT 0 COMMENT '预留数量',
    picked_quantity INT DEFAULT 0 COMMENT '已拣货数量',
    inbound_date TIMESTAMP COMMENT '入库日期',
    production_date DATE COMMENT '生产日期',
    expiry_date DATE COMMENT '失效日期',
    unit_cost DECIMAL(10,2) DEFAULT 0 COMMENT '单位成本',
    location_code VARCHAR(32) COMMENT '库位编码',
    status VARCHAR(32) DEFAULT 'AVAILABLE' COMMENT '状态：AVAILABLE/EXHAUSTED/LOCKED/EXPIRED',
    remark VARCHAR(512) COMMENT '备注',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    version INT DEFAULT 0 COMMENT '乐观锁版本',
    UNIQUE KEY uk_batch (sku_code, warehouse_id, batch_no),
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_batch_no (batch_no),
    INDEX idx_inbound_date (inbound_date),
    INDEX idx_expiry_date (expiry_date),
    INDEX idx_status (status)
) COMMENT '库存批次表';

-- 出库策略配置表
CREATE TABLE IF NOT EXISTS outbound_strategy (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    strategy_code VARCHAR(64) NOT NULL COMMENT '策略编码',
    strategy_name VARCHAR(128) NOT NULL COMMENT '策略名称',
    strategy_type VARCHAR(32) NOT NULL COMMENT '策略类型：FIFO/LIFO/SPECIFIC_BATCH',
    warehouse_id BIGINT COMMENT '仓库ID（空为全局策略）',
    warehouse_code VARCHAR(32) COMMENT '仓库编码',
    sku_code VARCHAR(64) COMMENT 'SKU编码（空为适用所有）',
    sku_code_pattern VARCHAR(128) COMMENT 'SKU编码匹配模式（正则）',
    priority INT DEFAULT 100 COMMENT '优先级（越小越高）',
    specific_batch_no VARCHAR(64) COMMENT '指定批次号（仅SPECIFIC_BATCH类型有效）',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    remark VARCHAR(512) COMMENT '备注',
    creator VARCHAR(64) COMMENT '创建人',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    version INT DEFAULT 0 COMMENT '乐观锁版本',
    UNIQUE KEY uk_strategy_code (strategy_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_sku_code (sku_code),
    INDEX idx_strategy_type (strategy_type),
    INDEX idx_is_active (is_active),
    INDEX idx_priority (priority)
) COMMENT '出库策略配置表';

-- 库存预警配置表
CREATE TABLE IF NOT EXISTS inventory_alert_config (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    config_code VARCHAR(64) NOT NULL COMMENT '配置编码',
    warehouse_code VARCHAR(32) COMMENT '仓库编码（空为全局配置）',
    sku_code VARCHAR(64) COMMENT 'SKU编码（空为适用所有）',
    safety_stock_threshold INT COMMENT '安全库存阈值',
    level VARCHAR(32) DEFAULT 'WARNING' COMMENT '预警级别：INFO/WARNING/ERROR/CRITICAL',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    cooldown_minutes INT DEFAULT 60 COMMENT '冷却时间（分钟）',
    notification_channels VARCHAR(128) COMMENT '通知渠道：EMAIL,SMS,IN_APP,DINGTALK',
    notify_users VARCHAR(512) COMMENT '通知用户（逗号分隔）',
    created_by VARCHAR(64) COMMENT '创建人',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_by VARCHAR(64) COMMENT '更新人',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    version INT DEFAULT 0 COMMENT '乐观锁版本',
    UNIQUE KEY uk_config_code (config_code),
    INDEX idx_warehouse_code (warehouse_code),
    INDEX idx_sku_code (sku_code),
    INDEX idx_enabled (enabled)
) COMMENT '库存预警配置表';

-- 库存预警记录表
CREATE TABLE IF NOT EXISTS inventory_alert (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    alert_code VARCHAR(64) NOT NULL COMMENT '预警编号',
    warehouse_code VARCHAR(32) NOT NULL COMMENT '仓库编码',
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    alert_type VARCHAR(32) NOT NULL COMMENT '预警类型：LOW_STOCK/OVERSTOCK/EXPIRING_SOON/EXPIRED',
    level VARCHAR(32) DEFAULT 'WARNING' COMMENT '预警级别：INFO/WARNING/ERROR/CRITICAL',
    title VARCHAR(256) NOT NULL COMMENT '标题',
    description VARCHAR(1024) COMMENT '描述',
    current_quantity INT COMMENT '当前数量',
    threshold_value INT COMMENT '阈值',
    status VARCHAR(32) DEFAULT 'TRIGGERED' COMMENT '状态：TRIGGERED/ACKNOWLEDGED/IN_PROGRESS/RESOLVED/CLOSED',
    solution VARCHAR(1024) COMMENT '解决方案',
    acknowledged_by VARCHAR(64) COMMENT '确认人',
    acknowledged_at TIMESTAMP COMMENT '确认时间',
    resolved_by VARCHAR(64) COMMENT '解决人',
    resolved_at TIMESTAMP COMMENT '解决时间',
    occurred_at TIMESTAMP NOT NULL COMMENT '发生时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    version INT DEFAULT 0 COMMENT '乐观锁版本',
    UNIQUE KEY uk_alert_code (alert_code),
    INDEX idx_warehouse_code (warehouse_code),
    INDEX idx_sku_code (sku_code),
    INDEX idx_alert_type (alert_type),
    INDEX idx_level (level),
    INDEX idx_status (status),
    INDEX idx_occurred_at (occurred_at),
    INDEX idx_created_at (created_at)
) COMMENT '库存预警记录表';

-- 库存预占记录表
CREATE TABLE IF NOT EXISTS inventory_reservation_record (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    biz_id VARCHAR(64) NOT NULL COMMENT '业务单据ID',
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    quantity INT NOT NULL COMMENT '预占数量',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT '状态：PENDING/CONFIRMED/CANCELLED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    UNIQUE KEY uk_biz_id (biz_id),
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_status (status)
) COMMENT '库存预占记录表';

-- 库存操作审计日志表
CREATE TABLE IF NOT EXISTS inventory_operation_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    operation_type VARCHAR(32) NOT NULL COMMENT '操作类型：RESERVE/CONFIRM/CANCEL/INCREASE/DECREASE',
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    warehouse_id BIGINT NOT NULL COMMENT '仓库ID',
    quantity INT NOT NULL COMMENT '变动数量(增为正，减为负)',
    biz_id VARCHAR(64) COMMENT '业务单据ID',
    remark VARCHAR(512) COMMENT '备注',
    operator_id VARCHAR(64) COMMENT '操作人ID',
    operator_name VARCHAR(128) COMMENT '操作人名称',
    quantity_before INT COMMENT '操作前库存',
    quantity_after INT COMMENT '操作后库存',
    operated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间',
    result VARCHAR(32) DEFAULT 'SUCCESS' COMMENT '操作结果：SUCCESS/FAILED',
    error_message VARCHAR(1024) COMMENT '错误信息',
    request_id VARCHAR(64) COMMENT '请求ID',
    client_ip VARCHAR(64) COMMENT '客户端IP',
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_operation_type (operation_type),
    INDEX idx_biz_id (biz_id),
    INDEX idx_operator_id (operator_id),
    INDEX idx_operated_at (operated_at),
    INDEX idx_result (result)
) COMMENT '库存操作审计日志表';