-- Finance Service Database Migration V1
-- Created: 2026-05-29
-- Description: Initialize finance module tables (account, account_subject, voucher, voucher_line)

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