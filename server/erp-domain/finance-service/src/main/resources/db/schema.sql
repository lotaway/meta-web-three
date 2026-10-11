-- Finance Service Database Schema
-- Combined from V1__finance_init.sql and V2__ar_ap_init.sql
-- Description: Initialize finance module tables

-- ============================================
-- Part 1: Finance Account and Voucher Tables
-- ============================================

-- Table: finance_account
CREATE TABLE IF NOT EXISTS finance_account (
    id BIGSERIAL PRIMARY KEY,
    account_no VARCHAR(50) NOT NULL UNIQUE,  -- Account number,
    account_name VARCHAR(100) NOT NULL,  -- Account name,
    type VARCHAR(20) NOT NULL,  -- Account type: CASH, BANK, VIRTUAL, CREDIT,
    balance DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Account balance,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',  -- Account status: ACTIVE, FROZEN, CLOSED,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Creation time,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Update time,
    version INT NOT NULL DEFAULT 0  -- Optimistic lock version
);
CREATE INDEX IF NOT EXISTS idx_account_no ON finance_account (account_no);
CREATE INDEX IF NOT EXISTS idx_status ON finance_account (status);
CREATE INDEX IF NOT EXISTS idx_type ON finance_account (type);

-- Table: finance_account_subject
CREATE TABLE IF NOT EXISTS finance_account_subject (
    id BIGSERIAL PRIMARY KEY,
    subject_code VARCHAR(50) NOT NULL UNIQUE,  -- Subject code,
    subject_name VARCHAR(100) NOT NULL,  -- Subject name,
    direction VARCHAR(10) NOT NULL,  -- Subject direction: DEBIT, CREDIT,
    parent_id BIGINT,  -- Parent subject ID for hierarchical structure,
    level INT NOT NULL DEFAULT 1,  -- Subject level (1=root, 2=child),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',  -- Subject status: ACTIVE, DISABLED,
    balance DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Subject balance,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Creation time,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Update time,
    version INT NOT NULL DEFAULT 0,  -- Optimistic lock version,
    FOREIGN KEY (parent_id) REFERENCES finance_account_subject(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_subject_code ON finance_account_subject (subject_code);
CREATE INDEX IF NOT EXISTS idx_parent_id ON finance_account_subject (parent_id);
CREATE INDEX IF NOT EXISTS idx_status ON finance_account_subject (status);

-- Table: finance_voucher
CREATE TABLE IF NOT EXISTS finance_voucher (
    id BIGSERIAL PRIMARY KEY,
    voucher_no VARCHAR(50) NOT NULL UNIQUE,  -- Voucher number,
    type VARCHAR(20) NOT NULL,  -- Voucher type: RECEIPT, PAYMENT, TRANSFER, GENERAL,
    voucher_date TIMESTAMP NOT NULL,  -- Voucher date,
    description VARCHAR(500),  -- Voucher description,
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',  -- Voucher status: DRAFT, PENDING_APPROVAL, APPROVED, POSTED, REJECTED,
    created_by VARCHAR(50) NOT NULL,  -- Creator username,
    approved_by VARCHAR(50),  -- Approver username,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Creation time,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Update time,
    version INT NOT NULL DEFAULT 0  -- Optimistic lock version
);
CREATE INDEX IF NOT EXISTS idx_voucher_no ON finance_voucher (voucher_no);
CREATE INDEX IF NOT EXISTS idx_status ON finance_voucher (status);
CREATE INDEX IF NOT EXISTS idx_type ON finance_voucher (type);
CREATE INDEX IF NOT EXISTS idx_voucher_date ON finance_voucher (voucher_date);
CREATE INDEX IF NOT EXISTS idx_created_by ON finance_voucher (created_by);

-- Table: finance_voucher_line
CREATE TABLE IF NOT EXISTS finance_voucher_line (
    id BIGSERIAL PRIMARY KEY,
    voucher_id BIGINT NOT NULL,  -- Voucher ID (foreign key),
    subject_id BIGINT NOT NULL,  -- Account subject ID,
    debit_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Debit amount,
    credit_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Credit amount,
    FOREIGN KEY (voucher_id) REFERENCES finance_voucher(id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES finance_account_subject(id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_voucher_id ON finance_voucher_line (voucher_id);
CREATE INDEX IF NOT EXISTS idx_subject_id ON finance_voucher_line (subject_id);

-- Insert default account subjects for common accounting
INSERT INTO finance_account_subject (subject_code, subject_name, direction, parent_id, level, status, balance) VALUES
('1001', 'Cash', 'DEBIT', NULL, 1, 'ACTIVE', 0.00),
('1002', 'Bank Account', 'DEBIT', NULL, 1, 'ACTIVE', 0.00),
('2001', 'Accounts Payable', 'CREDIT', NULL, 1, 'ACTIVE', 0.00),
('2002', 'Advance from Customers', 'CREDIT', NULL, 1, 'ACTIVE', 0.00),
('4001', 'Sales Revenue', 'CREDIT', NULL, 1, 'ACTIVE', 0.00),
('5001', 'Cost of Goods Sold', 'DEBIT', NULL, 1, 'ACTIVE', 0.00);

-- ============================================
-- Part 2: Accounts Receivable/Payable Tables
-- ============================================

-- Table: accounts_receivable
CREATE TABLE IF NOT EXISTS accounts_receivable (
    id BIGSERIAL PRIMARY KEY,
    ar_code VARCHAR(50) NOT NULL UNIQUE,  -- AR code,
    customer_id BIGINT NOT NULL,  -- Customer ID,
    customer_name VARCHAR(100) NOT NULL,  -- Customer name,
    business_type VARCHAR(50),  -- Business type: SALE, SERVICE, RENTAL, OTHER,
    related_document_type VARCHAR(50),  -- Related document type: ORDER, INVOICE, CONTRACT,
    related_document_no VARCHAR(50),  -- Related document number,
    amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Total amount,
    received_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Received amount,
    remaining_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Remaining amount,
    invoice_date DATE NOT NULL,  -- Invoice date,
    due_date DATE NOT NULL,  -- Due date,
    credit_term INT NOT NULL DEFAULT 30,  -- Credit term in days,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- Status: PENDING, PARTIAL_RECEIVED, RECEIVED, OVERDUE, WRITE_OFF,
    currency VARCHAR(10) NOT NULL DEFAULT 'CNY',  -- Currency code,
    exchange_rate DECIMAL(18, 6) NOT NULL DEFAULT 1.000000,  -- Exchange rate to base currency,
    original_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Original amount in original currency,
    description VARCHAR(500),  -- Description,
    created_by BIGINT NOT NULL,  -- Creator ID,
    creator_name VARCHAR(50),  -- Creator name,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Creation time,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Update time,
    is_active BOOLEAN NOT NULL DEFAULT TRUE  -- Is active
);
CREATE INDEX IF NOT EXISTS idx_ar_code ON accounts_receivable (ar_code);
CREATE INDEX IF NOT EXISTS idx_customer_id ON accounts_receivable (customer_id);
CREATE INDEX IF NOT EXISTS idx_status ON accounts_receivable (status);
CREATE INDEX IF NOT EXISTS idx_due_date ON accounts_receivable (due_date);
CREATE INDEX IF NOT EXISTS idx_invoice_date ON accounts_receivable (invoice_date);

-- Table: accounts_payable
CREATE TABLE IF NOT EXISTS accounts_payable (
    id BIGSERIAL PRIMARY KEY,
    ap_code VARCHAR(50) NOT NULL UNIQUE,  -- AP code,
    supplier_id BIGINT NOT NULL,  -- Supplier ID,
    supplier_name VARCHAR(100) NOT NULL,  -- Supplier name,
    business_type VARCHAR(50),  -- Business type: PURCHASE, SERVICE, RENTAL, OTHER,
    related_document_type VARCHAR(50),  -- Related document type: ORDER, INVOICE, CONTRACT,
    related_document_no VARCHAR(50),  -- Related document number,
    amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Total amount,
    paid_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Paid amount,
    remaining_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Remaining amount,
    invoice_date DATE NOT NULL,  -- Invoice date,
    due_date DATE NOT NULL,  -- Due date,
    credit_term INT NOT NULL DEFAULT 30,  -- Credit term in days,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- Status: PENDING, PARTIAL_PAID, PAID, OVERDUE, WRITE_OFF,
    currency VARCHAR(10) NOT NULL DEFAULT 'CNY',  -- Currency code,
    exchange_rate DECIMAL(18, 6) NOT NULL DEFAULT 1.000000,  -- Exchange rate to base currency,
    original_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,  -- Original amount in original currency,
    description VARCHAR(500),  -- Description,
    created_by BIGINT NOT NULL,  -- Creator ID,
    creator_name VARCHAR(50),  -- Creator name,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Creation time,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Update time,
    is_active BOOLEAN NOT NULL DEFAULT TRUE  -- Is active
);
CREATE INDEX IF NOT EXISTS idx_ap_code ON accounts_payable (ap_code);
CREATE INDEX IF NOT EXISTS idx_supplier_id ON accounts_payable (supplier_id);
CREATE INDEX IF NOT EXISTS idx_status ON accounts_payable (status);
CREATE INDEX IF NOT EXISTS idx_due_date ON accounts_payable (due_date);
CREATE INDEX IF NOT EXISTS idx_invoice_date ON accounts_payable (invoice_date);
-- ============================================
-- Part: Invoice tables (merged from invoice-service)
-- ============================================

CREATE TABLE IF NOT EXISTS invoice (
    id BIGSERIAL PRIMARY KEY,
    invoice_no VARCHAR(50) NOT NULL UNIQUE,
    order_no VARCHAR(50),
    customer_id BIGINT NOT NULL,
    customer_name VARCHAR(100) NOT NULL,
    customer_tax_no VARCHAR(50),
    customer_address VARCHAR(200),
    customer_bank VARCHAR(100),
    customer_account VARCHAR(50),
    type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',
    amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    tax_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    tax_rate VARCHAR(10),
    issue_date TIMESTAMP,
    issuer VARCHAR(50),
    remark VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_invoice_invoice_no ON invoice (invoice_no);
CREATE INDEX IF NOT EXISTS idx_invoice_order_no ON invoice (order_no);
CREATE INDEX IF NOT EXISTS idx_invoice_customer_id ON invoice (customer_id);
CREATE INDEX IF NOT EXISTS idx_invoice_status ON invoice (status);
CREATE INDEX IF NOT EXISTS idx_invoice_type ON invoice (type);
CREATE INDEX IF NOT EXISTS idx_invoice_issue_date ON invoice (issue_date);

-- ============================================
-- Part: Report tables (merged from reporting-service)
-- ============================================

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

-- ============================================
-- Part: Settlement tables (merged from settlement-service)
-- ============================================

-- Table: settlement_order
CREATE TABLE IF NOT EXISTS settlement_order (
    id BIGSERIAL PRIMARY KEY,
    settlement_no VARCHAR(50) NOT NULL UNIQUE,
    order_no VARCHAR(50) NOT NULL,
    merchant_id BIGINT NOT NULL,
    merchant_name VARCHAR(100) NOT NULL,
    order_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    settlement_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    commission_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    refund_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    channel VARCHAR(50),
    settlement_date TIMESTAMP,
    description VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_settlement_order_settlement_no ON settlement_order (settlement_no);
CREATE INDEX IF NOT EXISTS idx_settlement_order_no ON settlement_order (order_no);
CREATE INDEX IF NOT EXISTS idx_settlement_order_merchant_id ON settlement_order (merchant_id);
CREATE INDEX IF NOT EXISTS idx_settlement_order_status ON settlement_order (status);
CREATE INDEX IF NOT EXISTS idx_settlement_order_date ON settlement_order (settlement_date);

-- Table: reconciliation_record
CREATE TABLE IF NOT EXISTS reconciliation_record (
    id BIGSERIAL PRIMARY KEY,
    record_no VARCHAR(50) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL,
    reconcile_date TIMESTAMP NOT NULL,
    channel VARCHAR(50) NOT NULL,
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    total_count INT NOT NULL DEFAULT 0,
    matched_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    matched_count INT NOT NULL DEFAULT 0,
    unmatched_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    unmatched_count INT NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    remark VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reconciliation_record_no ON reconciliation_record (record_no);
CREATE INDEX IF NOT EXISTS idx_reconciliation_type ON reconciliation_record (type);
CREATE INDEX IF NOT EXISTS idx_reconciliation_channel ON reconciliation_record (channel);
CREATE INDEX IF NOT EXISTS idx_reconciliation_date ON reconciliation_record (reconcile_date);
CREATE INDEX IF NOT EXISTS idx_reconciliation_status ON reconciliation_record (status);

-- Table: split_rule
CREATE TABLE IF NOT EXISTS split_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_no VARCHAR(50) NOT NULL UNIQUE,
    rule_name VARCHAR(100) NOT NULL,
    type VARCHAR(20) NOT NULL,
    merchant_id BIGINT,
    ratio DECIMAL(5, 4),
    fixed_amount DECIMAL(18, 2),
    min_amount DECIMAL(18, 2),
    max_amount DECIMAL(18, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    priority INT NOT NULL DEFAULT 0,
    effective_date TIMESTAMP,
    expire_date TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_split_rule_no ON split_rule (rule_no);
CREATE INDEX IF NOT EXISTS idx_split_rule_merchant_id ON split_rule (merchant_id);
CREATE INDEX IF NOT EXISTS idx_split_rule_type ON split_rule (type);
CREATE INDEX IF NOT EXISTS idx_split_rule_status ON split_rule (status);
CREATE INDEX IF NOT EXISTS idx_split_rule_priority ON split_rule (priority);

-- Table: logistics_settlement
CREATE TABLE IF NOT EXISTS logistics_settlement (
    id BIGSERIAL PRIMARY KEY,
    settlement_no VARCHAR(50) NOT NULL UNIQUE,
    tracking_no VARCHAR(50) NOT NULL,
    order_no VARCHAR(50),
    carrier_id BIGINT NOT NULL,
    carrier_name VARCHAR(100) NOT NULL,
    freight DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    handling_fee DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    discount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    total_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    billing_cycle VARCHAR(20) NOT NULL DEFAULT 'MONTHLY',
    settlement_date TIMESTAMP,
    paid_at TIMESTAMP,
    remark VARCHAR(500),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT NOT NULL DEFAULT 0,
    deleted SMALLINT NOT NULL DEFAULT 0,
    UNIQUE (tracking_no, deleted)
);
CREATE INDEX IF NOT EXISTS idx_logistics_settlement_no ON logistics_settlement (settlement_no);
CREATE INDEX IF NOT EXISTS idx_logistics_tracking_no ON logistics_settlement (tracking_no);
CREATE INDEX IF NOT EXISTS idx_logistics_order_no ON logistics_settlement (order_no);
CREATE INDEX IF NOT EXISTS idx_logistics_carrier_id ON logistics_settlement (carrier_id);
CREATE INDEX IF NOT EXISTS idx_logistics_carrier_name ON logistics_settlement (carrier_name);
CREATE INDEX IF NOT EXISTS idx_logistics_status ON logistics_settlement (status);
CREATE INDEX IF NOT EXISTS idx_logistics_billing_cycle ON logistics_settlement (billing_cycle);
CREATE INDEX IF NOT EXISTS idx_logistics_settlement_date ON logistics_settlement (settlement_date);
CREATE INDEX IF NOT EXISTS idx_logistics_paid_at ON logistics_settlement (paid_at);
