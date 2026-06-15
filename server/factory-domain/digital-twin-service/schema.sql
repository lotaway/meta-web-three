-- Digital Twin Service - Database Schema Extension
-- Warehouse Management Module

-- Warehouses
CREATE TABLE IF NOT EXISTS warehouses (
    id BIGSERIAL PRIMARY KEY,
    warehouse_code VARCHAR(50) NOT NULL UNIQUE,
    warehouse_name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'OPERATING',
    total_area DECIMAL(10, 2),
    used_area DECIMAL(10, 2),
    location VARCHAR(200),
    center_x DECIMAL(10, 2),
    center_y DECIMAL(10, 2),
    center_z DECIMAL(10, 2),
    width DECIMAL(10, 2),
    length DECIMAL(10, 2),
    height DECIMAL(10, 2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_warehouse_status CHECK (status IN ('PLANNING', 'CONSTRUCTION', 'OPERATING', 'MAINTENANCE', 'DECOMMISSIONED'))
);

-- Shelves
CREATE TABLE IF NOT EXISTS shelves (
    id BIGSERIAL PRIMARY KEY,
    shelf_code VARCHAR(50) NOT NULL UNIQUE,
    warehouse_code VARCHAR(50) NOT NULL,
    zone VARCHAR(50),
    row_number INT NOT NULL,
    column_number INT NOT NULL,
    level_number INT NOT NULL DEFAULT 1,
    total_levels INT NOT NULL DEFAULT 3,
    status VARCHAR(20) NOT NULL DEFAULT 'EMPTY',
    max_weight DECIMAL(10, 2),
    current_weight DECIMAL(10, 2) DEFAULT 0,
    position_x DECIMAL(10, 2),
    position_y DECIMAL(10, 2),
    position_z DECIMAL(10, 2),
    rotation_y DECIMAL(10, 5) DEFAULT 0,
    length DECIMAL(10, 2),
    width DECIMAL(10, 2),
    height DECIMAL(10, 2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_shelf_status CHECK (status IN ('EMPTY', 'OCCUPIED', 'FULL', 'MAINTENANCE', 'OUT_OF_SERVICE'))
);

-- Inventory Items
CREATE TABLE IF NOT EXISTS inventory_items (
    id BIGSERIAL PRIMARY KEY,
    item_code VARCHAR(50) NOT NULL UNIQUE,
    sku VARCHAR(50) NOT NULL,
    item_name VARCHAR(200) NOT NULL,
    category VARCHAR(50),
    unit VARCHAR(20),
    quantity DECIMAL(12, 2) NOT NULL DEFAULT 0,
    min_quantity DECIMAL(12, 2) DEFAULT 0,
    max_quantity DECIMAL(12, 2),
    shelf_code VARCHAR(50),
    batch_number VARCHAR(50),
    production_date DATE,
    expiry_date DATE,
    unit_price DECIMAL(12, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'NORMAL',
    last_restock_date DATE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_inventory_status CHECK (status IN ('NORMAL', 'LOW', 'CRITICAL', 'EXPIRED', 'OUT_OF_STOCK'))
);

-- Inventory Alerts
CREATE TABLE IF NOT EXISTS inventory_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_code VARCHAR(50) NOT NULL UNIQUE,
    warehouse_code VARCHAR(50),
    shelf_code VARCHAR(50),
    item_code VARCHAR(50),
    alert_type VARCHAR(30) NOT NULL,
    level VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description VARCHAR(1000),
    current_quantity DECIMAL(12, 2),
    threshold_value DECIMAL(12, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'TRIGGERED',
    solution VARCHAR(1000),
    acknowledged_by VARCHAR(50),
    resolved_by VARCHAR(50),
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_inv_alert_type CHECK (alert_type IN ('LOW_STOCK', 'OVERSTOCK', 'EXPIRING_SOON', 'EXPIRED', 'SHELF_FULL', 'SHELF_EMPTY')),
    CONSTRAINT chk_inv_alert_level CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    CONSTRAINT chk_inv_alert_status CHECK (status IN ('TRIGGERED', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED'))
);

-- Inventory Movement Logs (-)
CREATE TABLE IF NOT EXISTS inventory_movement_logs (
    id BIGSERIAL PRIMARY KEY,
    item_code VARCHAR(50) NOT NULL,
    movement_type VARCHAR(20) NOT NULL,
    quantity_change DECIMAL(12, 2) NOT NULL,
    quantity_before DECIMAL(12, 2) NOT NULL,
    quantity_after DECIMAL(12, 2) NOT NULL,
    shelf_code VARCHAR(50),
    batch_number VARCHAR(50),
    operator_id VARCHAR(50),
    reference_number VARCHAR(100),
    remarks VARCHAR(500),
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_movement_type CHECK (movement_type IN ('INBOUND', 'OUTBOUND', 'TRANSFER', 'ADJUSTMENT', 'RETURN', 'SCRAP'))
);

-- Indexes for Warehouse Management
CREATE INDEX IF NOT EXISTS idx_warehouses_code ON warehouses(warehouse_code);
CREATE INDEX IF NOT EXISTS idx_warehouses_status ON warehouses(status);
CREATE INDEX IF NOT EXISTS idx_shelves_warehouse ON shelves(warehouse_code);
CREATE INDEX IF NOT EXISTS idx_shelves_code ON shelves(shelf_code);
CREATE INDEX IF NOT EXISTS idx_shelves_status ON shelves(status);
CREATE INDEX IF NOT EXISTS idx_shelves_location ON shelves(zone, row_number, column_number);
CREATE INDEX IF NOT EXISTS idx_inventory_items_sku ON inventory_items(sku);
CREATE INDEX IF NOT EXISTS idx_inventory_items_shelf ON inventory_items(shelf_code);
CREATE INDEX IF NOT EXISTS idx_inventory_items_status ON inventory_items(status);
CREATE INDEX IF NOT EXISTS idx_inventory_alerts_warehouse ON inventory_alerts(warehouse_code);
CREATE INDEX IF NOT EXISTS idx_inventory_alerts_item ON inventory_alerts(item_code);
CREATE INDEX IF NOT EXISTS idx_inventory_alerts_status ON inventory_alerts(status);
CREATE INDEX IF NOT EXISTS idx_inventory_alerts_level ON inventory_alerts(level);
CREATE INDEX IF NOT EXISTS idx_inventory_movement_logs_item ON inventory_movement_logs(item_code);
CREATE INDEX IF NOT EXISTS idx_inventory_movement_logs_time ON inventory_movement_logs(recorded_at);

-- Foreign Key Constraints
ALTER TABLE shelves ADD CONSTRAINT fk_shelf_warehouse 
    FOREIGN KEY (warehouse_code) REFERENCES warehouses(warehouse_code);

ALTER TABLE inventory_items ADD CONSTRAINT fk_inventory_shelf 
    FOREIGN KEY (shelf_code) REFERENCES shelves(shelf_code);

ALTER TABLE inventory_alerts ADD CONSTRAINT fk_inv_alert_warehouse 
    FOREIGN KEY (warehouse_code) REFERENCES warehouses(warehouse_code);

ALTER TABLE inventory_alerts ADD CONSTRAINT fk_inv_alert_shelf 
    FOREIGN KEY (shelf_code) REFERENCES shelves(shelf_code);

ALTER TABLE inventory_alerts ADD CONSTRAINT fk_inv_alert_item 
    FOREIGN KEY (item_code) REFERENCES inventory_items(item_code);

ALTER TABLE inventory_movement_logs ADD CONSTRAINT fk_movement_item 
    FOREIGN KEY (item_code) REFERENCES inventory_items(item_code);

ALTER TABLE inventory_movement_logs ADD CONSTRAINT fk_movement_shelf 
    FOREIGN KEY (shelf_code) REFERENCES shelves(shelf_code);