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
