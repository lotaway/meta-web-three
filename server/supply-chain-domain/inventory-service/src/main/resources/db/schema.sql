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