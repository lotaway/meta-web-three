
CREATE TABLE IF NOT EXISTS rp_sales_report (
    id BIGSERIAL PRIMARY KEY,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_report_no ON rp_sales_report (report_no);
CREATE INDEX IF NOT EXISTS idx_type ON rp_sales_report (type);
CREATE INDEX IF NOT EXISTS idx_report_date ON rp_sales_report (report_date);

CREATE TABLE IF NOT EXISTS rp_inventory_report (
    id BIGSERIAL PRIMARY KEY,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_report_no ON rp_inventory_report (report_no);
CREATE INDEX IF NOT EXISTS idx_type ON rp_inventory_report (type);
CREATE INDEX IF NOT EXISTS idx_report_date ON rp_inventory_report (report_date);

CREATE TABLE IF NOT EXISTS rp_financial_report (
    id BIGSERIAL PRIMARY KEY,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_report_no ON rp_financial_report (report_no);
CREATE INDEX IF NOT EXISTS idx_type ON rp_financial_report (type);
CREATE INDEX IF NOT EXISTS idx_report_date ON rp_financial_report (report_date);

CREATE TABLE IF NOT EXISTS report_subscription (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    user_name VARCHAR(128),
    report_type VARCHAR(32) NOT NULL,
    frequency VARCHAR(32) NOT NULL,
    cron_expression VARCHAR(64),
    channel VARCHAR(32) NOT NULL,
    recipient VARCHAR(256),
    webhook_url VARCHAR(512),
    enabled BOOLEAN DEFAULT TRUE,
    next_send_time TIMESTAMP,
    last_send_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_id ON report_subscription (user_id);
CREATE INDEX IF NOT EXISTS idx_report_type ON report_subscription (report_type);
CREATE INDEX IF NOT EXISTS idx_enabled ON report_subscription (enabled);
CREATE INDEX IF NOT EXISTS idx_next_send_time ON report_subscription (next_send_time);
