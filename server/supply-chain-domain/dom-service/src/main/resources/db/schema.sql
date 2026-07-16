-- dom-service

CREATE TABLE IF NOT EXISTS dom_order (
    id BIGSERIAL PRIMARY KEY,
    dom_order_no VARCHAR(64) NOT NULL,
    original_order_no VARCHAR(64),
    customer_id VARCHAR(64),
    customer_name VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    total_amount DECIMAL(18,2),
    currency VARCHAR(8) DEFAULT 'CNY',
    priority INT DEFAULT 0,
    sourcing_strategy VARCHAR(32),
    region VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_dom_order_no ON dom_order (dom_order_no);
CREATE INDEX IF NOT EXISTS idx_original_order_no ON dom_order (original_order_no);
CREATE INDEX IF NOT EXISTS idx_status ON dom_order (status);

CREATE TABLE IF NOT EXISTS dom_order_line (
    id BIGSERIAL PRIMARY KEY,
    dom_order_id BIGINT NOT NULL,
    sku_code VARCHAR(64) NOT NULL, -- SKU
    sku_name VARCHAR(256),
    quantity INT NOT NULL DEFAULT 0,
    fulfilled_quantity INT DEFAULT 0,
    warehouse_id BIGINT, -- ID
    warehouse_name VARCHAR(128),
    unit_price DECIMAL(18,2),
    status VARCHAR(32) DEFAULT 'PENDING',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dom_order_id ON dom_order_line (dom_order_id);
CREATE INDEX IF NOT EXISTS idx_sku_code ON dom_order_line (sku_code);

CREATE TABLE IF NOT EXISTS fulfillment_plan (
    id BIGSERIAL PRIMARY KEY,
    dom_order_id BIGINT NOT NULL,
    dom_order_no VARCHAR(64),
    total_lines INT DEFAULT 0,
    fulfilled_lines INT DEFAULT 0,
    partially_fulfilled_lines INT DEFAULT 0,
    unfulfilled_lines INT DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'DRAFT',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dom_order_id ON fulfillment_plan (dom_order_id);
CREATE INDEX IF NOT EXISTS idx_dom_order_no ON fulfillment_plan (dom_order_no);
CREATE INDEX IF NOT EXISTS idx_status ON fulfillment_plan (status);

CREATE TABLE IF NOT EXISTS sourcing_rule (
    id BIGSERIAL PRIMARY KEY,
    rule_name VARCHAR(128) NOT NULL,
    rule_type VARCHAR(32) NOT NULL,
    priority INT DEFAULT 0,
    warehouse_ids VARCHAR(512),
    region VARCHAR(64),
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rule_type ON sourcing_rule (rule_type);
CREATE INDEX IF NOT EXISTS idx_region ON sourcing_rule (region);
CREATE INDEX IF NOT EXISTS idx_enabled ON sourcing_rule (enabled);
