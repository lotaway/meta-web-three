-- ===== ai-domain\forecasting-service\src\main\resources\db\schema.sql =====
-- forecasting-service schema (PostgreSQL)

CREATE TABLE IF NOT EXISTS tb_sales_history (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64),
    warehouse_id BIGINT,
    sales_date DATE,
    quantity INT,
    sales_channel VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sku_warehouse ON tb_sales_history (sku_code, warehouse_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON tb_sales_history (sales_date);
COMMENT ON TABLE tb_sales_history IS 'Sales history data for forecast model training';

CREATE TABLE IF NOT EXISTS tb_sales_forecast (
    id BIGSERIAL PRIMARY KEY,
    sku_code VARCHAR(64),
    sku_name VARCHAR(128),
    warehouse_id BIGINT,
    forecast_date DATE,
    forecast_quantity INT,
    actual_quantity INT,
    forecast_amount DECIMAL(18,2),
    actual_amount DECIMAL(18,2),
    status VARCHAR(32) DEFAULT 'PENDING',
    forecast_model VARCHAR(64),
    confidence_level DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sku_code ON tb_sales_forecast (sku_code);
CREATE INDEX IF NOT EXISTS idx_warehouse_id ON tb_sales_forecast (warehouse_id);
CREATE INDEX IF NOT EXISTS idx_forecast_date ON tb_sales_forecast (forecast_date);
CREATE INDEX IF NOT EXISTS idx_status ON tb_sales_forecast (status);
CREATE INDEX IF NOT EXISTS idx_sku_forecast_date ON tb_sales_forecast (sku_code, forecast_date);
COMMENT ON TABLE tb_sales_forecast IS 'Sales forecast data';

CREATE TABLE IF NOT EXISTS tb_forecast_model (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(128),
    model_type VARCHAR(64),
    model_version VARCHAR(32),
    status VARCHAR(32) DEFAULT 'DRAFT',
    accuracy DECIMAL(5,2),
    training_days INT,
    feature_config TEXT,
    algorithm VARCHAR(64),
    trained_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_model_name ON tb_forecast_model (model_name);
CREATE INDEX IF NOT EXISTS idx_model_type ON tb_forecast_model (model_type);
CREATE INDEX IF NOT EXISTS idx_model_status ON tb_forecast_model (status);
CREATE INDEX IF NOT EXISTS idx_model_type_status ON tb_forecast_model (model_type, status);
COMMENT ON TABLE tb_forecast_model IS 'Forecast model configuration';


-- ===== ai-domain\route-optimizer\src\main\resources\db\schema.sql =====
CREATE TABLE IF NOT EXISTS tb_vehicle (
    id              BIGSERIAL PRIMARY KEY,
    vehicle_code    VARCHAR(50)  NOT NULL UNIQUE,
    vehicle_number  VARCHAR(50)  NOT NULL UNIQUE,
    vehicle_type    VARCHAR(50),
    status          VARCHAR(20)  NOT NULL DEFAULT 'IDLE',
    max_load_capacity DECIMAL(10,2),
    current_load    DECIMAL(10,2) DEFAULT 0,
    fuel_capacity   DECIMAL(10,2),
    current_fuel    DECIMAL(10,2) DEFAULT 0,
    fuel_efficiency DECIMAL(10,4),
    driver_name     VARCHAR(50),
    driver_phone    VARCHAR(20),
    latitude        DECIMAL(10,6),
    longitude       DECIMAL(10,6),
    last_location_update TIMESTAMP,
    current_route_plan_code VARCHAR(50),
    total_deliveries INT DEFAULT 0,
    total_distance  DECIMAL(10,2) DEFAULT 0,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_vehicle_status ON tb_vehicle (status);
CREATE INDEX IF NOT EXISTS idx_vehicle_code ON tb_vehicle (vehicle_code);
CREATE INDEX IF NOT EXISTS idx_vehicle_number ON tb_vehicle (vehicle_number);
COMMENT ON TABLE tb_vehicle IS 'Vehicle information';

CREATE TABLE IF NOT EXISTS tb_route_plan (
    id                BIGSERIAL PRIMARY KEY,
    plan_code         VARCHAR(50)  NOT NULL UNIQUE,
    plan_name         VARCHAR(100),
    vehicle_code      VARCHAR(50),
    driver_name       VARCHAR(50),
    driver_phone      VARCHAR(20),
    status            VARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    total_distance    DECIMAL(10,2),
    estimated_duration INT,
    planned_start_time TIMESTAMP,
    planned_end_time   TIMESTAMP,
    actual_start_time  TIMESTAMP,
    actual_end_time    TIMESTAMP,
    optimization_type  VARCHAR(20),
    total_cost         DECIMAL(10,2),
    remarks           VARCHAR(500),
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_route_plan_status ON tb_route_plan (status);
CREATE INDEX IF NOT EXISTS idx_route_plan_code ON tb_route_plan (plan_code);
CREATE INDEX IF NOT EXISTS idx_route_plan_vehicle ON tb_route_plan (vehicle_code);
CREATE INDEX IF NOT EXISTS idx_route_plan_planned_start ON tb_route_plan (planned_start_time);
COMMENT ON TABLE tb_route_plan IS 'Route plan';

CREATE TABLE IF NOT EXISTS tb_route_point (
    id                      BIGSERIAL PRIMARY KEY,
    route_plan_id           BIGINT,
    point_code              VARCHAR(50)  NOT NULL UNIQUE,
    point_name              VARCHAR(100),
    type                    VARCHAR(20)  NOT NULL,
    latitude                DECIMAL(10,6),
    longitude               DECIMAL(10,6),
    address                 VARCHAR(200),
    contact_person          VARCHAR(50),
    contact_phone           VARCHAR(20),
    sequence                INT,
    estimated_arrival_time  DECIMAL(10,2),
    actual_arrival_time     DECIMAL(10,2),
    expected_service_duration INT,
    actual_service_duration   INT,
    status                  VARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    distance_from_previous  DECIMAL(10,2),
    order_code              VARCHAR(50),
    remarks                 VARCHAR(500),
    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_route_point_plan FOREIGN KEY (route_plan_id) REFERENCES tb_route_plan(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_route_point_plan ON tb_route_point (route_plan_id);
CREATE INDEX IF NOT EXISTS idx_route_point_code ON tb_route_point (point_code);
CREATE INDEX IF NOT EXISTS idx_route_point_status ON tb_route_point (status);
COMMENT ON TABLE tb_route_point IS 'Route points';


