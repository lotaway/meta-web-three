CREATE TABLE commission_relation (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    parent_user_id BIGINT NOT NULL,
    depth INT NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_commission_relation_user UNIQUE (user_id)
);

CREATE INDEX idx_commission_relation_parent ON commission_relation(parent_user_id);

CREATE TABLE commission_config (
    id BIGINT PRIMARY KEY,
    buy_rate DECIMAL(10,4) NOT NULL,
    level_rates VARCHAR(255) NOT NULL,
    max_levels INT NOT NULL,
    return_window_days INT NOT NULL,
    confirm_method VARCHAR(20) NOT NULL DEFAULT 'RECEIVE',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE commission_record (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    from_user_id BIGINT NOT NULL,
    level INT NOT NULL,
    amount DECIMAL(18,4) NOT NULL,
    status VARCHAR(20) NOT NULL,
    available_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_commission_record_order ON commission_record(order_id);
CREATE INDEX idx_commission_record_user ON commission_record(user_id);
CREATE INDEX idx_commission_record_available ON commission_record(available_at);

CREATE TABLE commission_main (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    total_amount DECIMAL(18,4) NOT NULL DEFAULT 0,
    available_amount DECIMAL(18,4) NOT NULL DEFAULT 0,
    frozen_amount DECIMAL(18,4) NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_commission_main_user UNIQUE (user_id)
);
