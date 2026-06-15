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