-- 仓库质检模块数据库表结构

-- 质检标准表：定义每个SKU的质检规则
CREATE TABLE IF NOT EXISTS quality_standard (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    product_name VARCHAR(255) COMMENT '产品名称',
    inspection_type VARCHAR(32) NOT NULL COMMENT '质检类型：FULL-全检, SAMPLE-抽检, AUTO-自动质检',
    inspection_level VARCHAR(32) NOT NULL COMMENT '质检等级：NORMAL-普通, STRICT-严格',
    sample_rate DECIMAL(5,2) DEFAULT 100.00 COMMENT '抽检比例(%)',
    check_items TEXT COMMENT '检查项目，JSON格式',
    acceptance_qty INTEGER DEFAULT 0 COMMENT '接收数量标准',
    defect_qty_threshold INTEGER DEFAULT 0 COMMENT '不良数量阈值',
    weight_tolerance DECIMAL(5,2) DEFAULT 0.00 COMMENT '重量允差(%)',
    dimension_tolerance VARCHAR(64) COMMENT '尺寸允差',
    packaging_requirement VARCHAR(512) COMMENT '包装要求',
    label_requirement VARCHAR(512) COMMENT '标签要求',
    is_active TINYINT DEFAULT 1 COMMENT '是否启用',
    remark VARCHAR(512) COMMENT '备注',
    creator VARCHAR(64) COMMENT '创建人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64) COMMENT '更新人',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted TINYINT DEFAULT 0,
    UNIQUE KEY uk_sku (sku_code, deleted)
) COMMENT '质检标准表';

-- 质检记录表：记录每次质检的结果
CREATE TABLE IF NOT EXISTS quality_inspection (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    inspection_no VARCHAR(64) NOT NULL COMMENT '质检单号',
    order_id BIGINT COMMENT '关联入库单ID',
    order_no VARCHAR(64) COMMENT '关联入库单号',
    inbound_type VARCHAR(32) COMMENT '入库类型',
    warehouse_id BIGINT COMMENT '仓库ID',
    supplier_code VARCHAR(64) COMMENT '供应商编码',
    supplier_name VARCHAR(255) COMMENT '供应商名称',
    inspection_type VARCHAR(32) NOT NULL COMMENT '质检类型：FULL-全检, SAMPLE-抽检',
    inspection_status VARCHAR(32) NOT NULL COMMENT '质检状态：PENDING-待质检, IN_PROGRESS-质检中, PASSED-合格, FAILED-不合格, CONCESSION-让步接收',
    total_quantity INTEGER DEFAULT 0 COMMENT '总数量',
    inspected_quantity INTEGER DEFAULT 0 COMMENT '已检数量',
    qualified_quantity INTEGER DEFAULT 0 COMMENT '合格数量',
    unqualified_quantity INTEGER DEFAULT 0 COMMENT '不合格数量',
    concession_quantity INTEGER DEFAULT 0 COMMENT '让步接收数量',
    defect_rate DECIMAL(5,2) DEFAULT 0.00 COMMENT '不良率(%)',
    inspector VARCHAR(64) COMMENT '质检员',
    inspection_time DATETIME COMMENT '质检时间',
    result_remark VARCHAR(512) COMMENT '结果备注',
    is_auto_inspection TINYINT DEFAULT 0 COMMENT '是否自动质检',
    source_system VARCHAR(64) COMMENT '来源系统',
    creator VARCHAR(64) COMMENT '创建人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64) COMMENT '更新人',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted TINYINT DEFAULT 0,
    UNIQUE KEY uk_inspection_no (inspection_no, deleted)
) COMMENT '质检记录表';

-- 质检明细表：记录每个SKU的质检详情
CREATE TABLE IF NOT EXISTS quality_inspection_item (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    inspection_id BIGINT NOT NULL COMMENT '质检记录ID',
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    product_name VARCHAR(255) COMMENT '产品名称',
    batch_no VARCHAR(64) COMMENT '批次号',
    location_code VARCHAR(64) COMMENT '库位编码',
    plan_quantity INTEGER DEFAULT 0 COMMENT '计划数量',
    actual_quantity INTEGER DEFAULT 0 COMMENT '实际数量',
    inspected_quantity INTEGER DEFAULT 0 COMMENT '已检数量',
    qualified_quantity INTEGER DEFAULT 0 COMMENT '合格数量',
    unqualified_quantity INTEGER DEFAULT 0 COMMENT '不合格数量',
    concession_quantity INTEGER DEFAULT 0 COMMENT '让步接收数量',
    sample_quantity INTEGER DEFAULT 0 COMMENT '抽检数量',
    defect_items TEXT COMMENT '不良项目，JSON格式',
    check_result VARCHAR(32) NOT NULL COMMENT '检查结果：QUALIFIED-合格, UNQUALIFIED-不合格, CONCESSION-让步接收',
    remark VARCHAR(512) COMMENT '备注',
    creator VARCHAR(64) COMMENT '创建人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64) COMMENT '更新人',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted TINYINT DEFAULT 0,
    KEY idx_inspection_id (inspection_id, deleted),
    KEY idx_sku (sku_code, deleted)
) COMMENT '质检明细表';

-- 不良品记录表：记录每个不良品项
CREATE TABLE IF NOT EXISTS defect_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    inspection_id BIGINT NOT NULL COMMENT '质检记录ID',
    inspection_item_id BIGINT COMMENT '质检明细ID',
    sku_code VARCHAR(64) NOT NULL COMMENT 'SKU编码',
    product_name VARCHAR(255) COMMENT '产品名称',
    batch_no VARCHAR(64) COMMENT '批次号',
    defect_type VARCHAR(64) NOT NULL COMMENT '不良类型：外观缺陷, 数量短缺, 包装破损, 标签错误, 规格不符, 过期, 其他',
    defect_name VARCHAR(128) NOT NULL COMMENT '不良名称',
    defect_description VARCHAR(512) COMMENT '不良描述',
    defect_quantity INTEGER DEFAULT 0 COMMENT '不良数量',
    defect_level VARCHAR(32) NOT NULL COMMENT '不良等级：CRITICAL-严重, MAJOR-主要, MINOR-次要',
    photo_urls TEXT COMMENT '不良照片，JSON数组',
    location_code VARCHAR(64) COMMENT '库位编码',
    creator VARCHAR(64) COMMENT '创建人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64) COMMENT '更新人',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted TINYINT DEFAULT 0,
    KEY idx_inspection_id (inspection_id, deleted),
    KEY idx_sku (sku_code, deleted)
) COMMENT '不良品记录表';

-- 不良品处理记录表：记录不良品的处理方式
CREATE TABLE IF NOT EXISTS defect_processing (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    defect_id BIGINT NOT NULL COMMENT '不良品记录ID',
    processing_no VARCHAR(64) NOT NULL COMMENT '处理单号',
    processing_type VARCHAR(32) NOT NULL COMMENT '处理类型：RETURN-退货, EXCHANGE-换货, DISCOUNT-折价, SCRAP-报废, SPECIAL_USE-特采',
    processing_status VARCHAR(32) NOT NULL COMMENT '处理状态：PENDING-待处理, PROCESSING-处理中, COMPLETED-已完成, CANCELLED-已取消',
    processing_quantity INTEGER DEFAULT 0 COMMENT '处理数量',
    processing_price DECIMAL(12,2) DEFAULT 0.00 COMMENT '处理价格/折价金额',
    processing_reason VARCHAR(512) COMMENT '处理原因',
    processing_remark VARCHAR(512) COMMENT '处理备注',
    processor VARCHAR(64) COMMENT '处理人',
    processing_time DATETIME COMMENT '处理时间',
    related_document_no VARCHAR(64) COMMENT '关联单据号',
    related_document_type VARCHAR(32) COMMENT '关联单据类型',
    approver VARCHAR(64) COMMENT '审批人',
    approve_time DATETIME COMMENT '审批时间',
    approve_remark VARCHAR(512) COMMENT '审批备注',
    creator VARCHAR(64) COMMENT '创建人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64) COMMENT '更新人',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted TINYINT DEFAULT 0,
    UNIQUE KEY uk_processing_no (processing_no, deleted),
    KEY idx_defect_id (defect_id, deleted)
) COMMENT '不良品处理记录表';

-- 质检配置表：系统级质检配置
CREATE TABLE IF NOT EXISTS quality_config (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(64) NOT NULL COMMENT '配置键',
    config_value TEXT COMMENT '配置值',
    config_type VARCHAR(32) NOT NULL COMMENT '配置类型：SYSTEM-系统配置, WAREHOUSE-仓库配置',
    warehouse_id BIGINT COMMENT '仓库ID',
    description VARCHAR(255) COMMENT '描述',
    is_active TINYINT DEFAULT 1 COMMENT '是否启用',
    creator VARCHAR(64) COMMENT '创建人',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updater VARCHAR(64) COMMENT '更新人',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted TINYINT DEFAULT 0,
    UNIQUE KEY uk_key_warehouse (config_key, warehouse_id, deleted)
) COMMENT '质检配置表';

-- 索引优化
CREATE INDEX idx_quality_standard_sku ON quality_standard(sku_code);
CREATE INDEX idx_quality_inspection_order ON quality_inspection(order_id);
CREATE INDEX idx_quality_inspection_status ON quality_inspection(inspection_status);
CREATE INDEX idx_defect_record_type ON defect_record(defect_type);
CREATE INDEX idx_defect_processing_status ON defect_processing(processing_status);