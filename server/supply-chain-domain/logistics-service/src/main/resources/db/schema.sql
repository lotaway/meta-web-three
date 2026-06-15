-- logistics-service

CREATE TABLE IF NOT EXISTS logistics_carrier (
    id BIGSERIAL PRIMARY KEY,
    carrier_code VARCHAR(64) NOT NULL UNIQUE,
    carrier_name VARCHAR(128) NOT NULL,
 carrier_type VARCHAR(32), -- EXPRESS/FREIGHT/POST,
    contact VARCHAR(64),
    phone VARCHAR(32),
    website VARCHAR(256),
 status VARCHAR(32) DEFAULT 'ACTIVE', -- ACTIVE/INACTIVE,
    base_freight DECIMAL(10,2) DEFAULT 0,
 weight_unit_price DECIMAL(10,4) DEFAULT 0, -- (/),
 volume_unit_price DECIMAL(10,4) DEFAULT 0, -- (/),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_carrier_code ON logistics_carrier (carrier_code);
CREATE INDEX IF NOT EXISTS idx_status ON logistics_carrier (status);

CREATE TABLE IF NOT EXISTS logistics_order (
    id BIGSERIAL PRIMARY KEY,
    tracking_no VARCHAR(64) NOT NULL UNIQUE,
    order_no VARCHAR(64) NOT NULL,
 carrier_id BIGINT, -- ID,
    carrier_name VARCHAR(128),
    service_type VARCHAR(32),
    sender_name VARCHAR(64) NOT NULL,
    sender_phone VARCHAR(32) NOT NULL,
    sender_province VARCHAR(32),
    sender_city VARCHAR(32),
    sender_district VARCHAR(32),
    sender_address VARCHAR(256),
    receiver_name VARCHAR(64) NOT NULL,
    receiver_phone VARCHAR(32) NOT NULL,
    receiver_province VARCHAR(32),
    receiver_city VARCHAR(32),
    receiver_district VARCHAR(32),
    receiver_address VARCHAR(256),
 weight DECIMAL(10,3), -- (kg),
    volume DECIMAL(10,3),
    freight DECIMAL(10,2) DEFAULT 0,
 status VARCHAR(32) DEFAULT 'PENDING', -- PENDING/PICKED_UP/IN_TRANSIT/OUT_FOR_DELIVERY/DELIVERED/EXCEPTION,
    picked_up_at TIMESTAMP,
    in_transit_at TIMESTAMP,
    out_for_delivery_at TIMESTAMP,
    delivered_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tracking_no ON logistics_order (tracking_no);
CREATE INDEX IF NOT EXISTS idx_order_no ON logistics_order (order_no);
CREATE INDEX IF NOT EXISTS idx_carrier_id ON logistics_order (carrier_id);
CREATE INDEX IF NOT EXISTS idx_status ON logistics_order (status);

CREATE TABLE IF NOT EXISTS logistics_tracking_event (
    id BIGSERIAL PRIMARY KEY,
    tracking_no VARCHAR(64) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    location VARCHAR(128),
    description VARCHAR(512),
    operator VARCHAR(64),
    occurred_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tracking_no ON logistics_tracking_event (tracking_no);
CREATE INDEX IF NOT EXISTS idx_occurred_at ON logistics_tracking_event (occurred_at);
