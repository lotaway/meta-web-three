-- 报表服务数据库初始化
-- 销售报表、库存报表、财务报表

-- 销售报表表
CREATE TABLE IF NOT EXISTS rp_sales_report (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    report_no VARCHAR(64) NOT NULL UNIQUE,
    type VARCHAR(32) NOT NULL,
    report_date TIMESTAMP,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    total_sales_amount DECIMAL(18, 2) DEFAULT 0,
    total_order_count INT DEFAULT 0,
    average_order_amount DECIMAL(18, 2) DEFAULT 0,
    gross_profit DECIMAL(18, 2) DEFAULT 0,
    profit_margin DECIMAL(5, 2) DEFAULT 0,
    category_breakdown TEXT,
    product_ranking TEXT,
    channel_breakdown TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_report_no (report_no),
    INDEX idx_type (type),
    INDEX idx_report_date (report_date)
);

-- 库存报表表
CREATE TABLE IF NOT EXISTS rp_inventory_report (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    report_no VARCHAR(64) NOT NULL UNIQUE,
    type VARCHAR(32) NOT NULL,
    report_date TIMESTAMP,
    total_inventory_value DECIMAL(18, 2) DEFAULT 0,
    total_sku_count INT DEFAULT 0,
    total_quantity INT DEFAULT 0,
    turnover_rate DECIMAL(5, 2) DEFAULT 0,
    slow_moving_rate DECIMAL(5, 2) DEFAULT 0,
    slow_moving_count INT DEFAULT 0,
    warehouse_breakdown TEXT,
    category_breakdown TEXT,
    low_stock_items TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_report_no (report_no),
    INDEX idx_type (type),
    INDEX idx_report_date (report_date)
);

-- 财务报表表
CREATE TABLE IF NOT EXISTS rp_financial_report (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    report_no VARCHAR(64) NOT NULL UNIQUE,
    type VARCHAR(32) NOT NULL,
    report_date TIMESTAMP,
    total_receivable DECIMAL(18, 2) DEFAULT 0,
    total_payable DECIMAL(18, 2) DEFAULT 0,
    net_receivable DECIMAL(18, 2) DEFAULT 0,
    aging_analysis TEXT,
    current_assets DECIMAL(18, 2) DEFAULT 0,
    current_liabilities DECIMAL(18, 2) DEFAULT 0,
    working_capital DECIMAL(18, 2) DEFAULT 0,
    current_ratio DECIMAL(5, 2) DEFAULT 0,
    receivables_by_customer TEXT,
    payables_by_supplier TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_report_no (report_no),
    INDEX idx_type (type),
    INDEX idx_report_date (report_date)
);
