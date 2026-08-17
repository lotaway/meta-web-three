-- Invoice Service Database Schema

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