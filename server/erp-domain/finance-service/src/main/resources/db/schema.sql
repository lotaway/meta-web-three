-- Finance Service Database Schema
-- Combined from V1__finance_init.sql and V2__ar_ap_init.sql
-- Description: Initialize finance module tables

-- ============================================
-- Part 1: Finance Account and Voucher Tables
-- ============================================

-- Table: finance_account
CREATE TABLE IF NOT EXISTS finance_account (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    account_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Account number',
    account_name VARCHAR(100) NOT NULL COMMENT 'Account name',
    type VARCHAR(20) NOT NULL COMMENT 'Account type: CASH, BANK, VIRTUAL, CREDIT',
    balance DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Account balance',
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' COMMENT 'Account status: ACTIVE, FROZEN, CLOSED',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_account_no (account_no),
    INDEX idx_status (status),
    INDEX idx_type (type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Finance account table';

-- Table: finance_account_subject
CREATE TABLE IF NOT EXISTS finance_account_subject (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    subject_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'Subject code',
    subject_name VARCHAR(100) NOT NULL COMMENT 'Subject name',
    direction VARCHAR(10) NOT NULL COMMENT 'Subject direction: DEBIT, CREDIT',
    parent_id BIGINT COMMENT 'Parent subject ID for hierarchical structure',
    level INT NOT NULL DEFAULT 1 COMMENT 'Subject level (1=root, 2=child)',
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' COMMENT 'Subject status: ACTIVE, DISABLED',
    balance DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Subject balance',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_subject_code (subject_code),
    INDEX idx_parent_id (parent_id),
    INDEX idx_status (status),
    FOREIGN KEY (parent_id) REFERENCES finance_account_subject(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Finance account subject table';

-- Table: finance_voucher
CREATE TABLE IF NOT EXISTS finance_voucher (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    voucher_no VARCHAR(50) NOT NULL UNIQUE COMMENT 'Voucher number',
    type VARCHAR(20) NOT NULL COMMENT 'Voucher type: RECEIPT, PAYMENT, TRANSFER, GENERAL',
    voucher_date TIMESTAMP NOT NULL COMMENT 'Voucher date',
    description VARCHAR(500) COMMENT 'Voucher description',
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT' COMMENT 'Voucher status: DRAFT, PENDING_APPROVAL, APPROVED, POSTED, REJECTED',
    created_by VARCHAR(50) NOT NULL COMMENT 'Creator username',
    approved_by VARCHAR(50) COMMENT 'Approver username',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    version INT NOT NULL DEFAULT 0 COMMENT 'Optimistic lock version',
    INDEX idx_voucher_no (voucher_no),
    INDEX idx_status (status),
    INDEX idx_type (type),
    INDEX idx_voucher_date (voucher_date),
    INDEX idx_created_by (created_by)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Finance voucher table';

-- Table: finance_voucher_line
CREATE TABLE IF NOT EXISTS finance_voucher_line (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    voucher_id BIGINT NOT NULL COMMENT 'Voucher ID (foreign key)',
    subject_id BIGINT NOT NULL COMMENT 'Account subject ID',
    debit_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Debit amount',
    credit_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Credit amount',
    INDEX idx_voucher_id (voucher_id),
    INDEX idx_subject_id (subject_id),
    FOREIGN KEY (voucher_id) REFERENCES finance_voucher(id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES finance_account_subject(id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Finance voucher line table';

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
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ar_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'AR code',
    customer_id BIGINT NOT NULL COMMENT 'Customer ID',
    customer_name VARCHAR(100) NOT NULL COMMENT 'Customer name',
    business_type VARCHAR(50) COMMENT 'Business type: SALE, SERVICE, RENTAL, OTHER',
    related_document_type VARCHAR(50) COMMENT 'Related document type: ORDER, INVOICE, CONTRACT',
    related_document_no VARCHAR(50) COMMENT 'Related document number',
    amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Total amount',
    received_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Received amount',
    remaining_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Remaining amount',
    invoice_date DATE NOT NULL COMMENT 'Invoice date',
    due_date DATE NOT NULL COMMENT 'Due date',
    credit_term INT NOT NULL DEFAULT 30 COMMENT 'Credit term in days',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT 'Status: PENDING, PARTIAL_RECEIVED, RECEIVED, OVERDUE, WRITE_OFF',
    currency VARCHAR(10) NOT NULL DEFAULT 'CNY' COMMENT 'Currency code',
    exchange_rate DECIMAL(18, 6) NOT NULL DEFAULT 1.000000 COMMENT 'Exchange rate to base currency',
    original_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Original amount in original currency',
    description VARCHAR(500) COMMENT 'Description',
    created_by BIGINT NOT NULL COMMENT 'Creator ID',
    creator_name VARCHAR(50) COMMENT 'Creator name',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    is_active BOOLEAN NOT NULL DEFAULT TRUE COMMENT 'Is active',
    INDEX idx_ar_code (ar_code),
    INDEX idx_customer_id (customer_id),
    INDEX idx_status (status),
    INDEX idx_due_date (due_date),
    INDEX idx_invoice_date (invoice_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Accounts receivable table';

-- Table: accounts_payable
CREATE TABLE IF NOT EXISTS accounts_payable (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ap_code VARCHAR(50) NOT NULL UNIQUE COMMENT 'AP code',
    supplier_id BIGINT NOT NULL COMMENT 'Supplier ID',
    supplier_name VARCHAR(100) NOT NULL COMMENT 'Supplier name',
    business_type VARCHAR(50) COMMENT 'Business type: PURCHASE, SERVICE, RENTAL, OTHER',
    related_document_type VARCHAR(50) COMMENT 'Related document type: ORDER, INVOICE, CONTRACT',
    related_document_no VARCHAR(50) COMMENT 'Related document number',
    amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Total amount',
    paid_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Paid amount',
    remaining_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Remaining amount',
    invoice_date DATE NOT NULL COMMENT 'Invoice date',
    due_date DATE NOT NULL COMMENT 'Due date',
    credit_term INT NOT NULL DEFAULT 30 COMMENT 'Credit term in days',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' COMMENT 'Status: PENDING, PARTIAL_PAID, PAID, OVERDUE, WRITE_OFF',
    currency VARCHAR(10) NOT NULL DEFAULT 'CNY' COMMENT 'Currency code',
    exchange_rate DECIMAL(18, 6) NOT NULL DEFAULT 1.000000 COMMENT 'Exchange rate to base currency',
    original_amount DECIMAL(18, 2) NOT NULL DEFAULT 0.00 COMMENT 'Original amount in original currency',
    description VARCHAR(500) COMMENT 'Description',
    created_by BIGINT NOT NULL COMMENT 'Creator ID',
    creator_name VARCHAR(50) COMMENT 'Creator name',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation time',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update time',
    is_active BOOLEAN NOT NULL DEFAULT TRUE COMMENT 'Is active',
    INDEX idx_ap_code (ap_code),
    INDEX idx_supplier_id (supplier_id),
    INDEX idx_status (status),
    INDEX idx_due_date (due_date),
    INDEX idx_invoice_date (invoice_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Accounts payable table';