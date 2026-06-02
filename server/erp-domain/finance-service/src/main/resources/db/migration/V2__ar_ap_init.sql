-- Finance Service Database Migration V2
-- Created: 2026-06-02
-- Description: Add AR/AP (Accounts Receivable/Payable) tables

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