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
