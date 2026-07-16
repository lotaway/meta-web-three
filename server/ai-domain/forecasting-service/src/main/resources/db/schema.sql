-- forecasting-service schema

CREATE TABLE IF NOT EXISTS tb_sales_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sku_code VARCHAR(64) COMMENT 'SKU code',
    warehouse_id BIGINT COMMENT 'Warehouse ID',
    sales_date DATE COMMENT 'Sales date',
    quantity INT COMMENT 'Sales quantity',
    sales_channel VARCHAR(32) COMMENT 'Sales channel',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sku_warehouse (sku_code, warehouse_id),
    INDEX idx_sales_date (sales_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Sales history data for forecast model training';

CREATE TABLE IF NOT EXISTS tb_sales_forecast (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sku_code VARCHAR(64) COMMENT 'SKU code',
    sku_name VARCHAR(128) COMMENT 'SKU name',
    warehouse_id BIGINT COMMENT 'Warehouse ID',
    forecast_date DATE COMMENT 'Forecast date',
    forecast_quantity INT COMMENT 'Forecast quantity',
    actual_quantity INT COMMENT 'Actual quantity',
    forecast_amount DECIMAL(18,2) COMMENT 'Forecast amount',
    actual_amount DECIMAL(18,2) COMMENT 'Actual amount',
    status VARCHAR(32) DEFAULT 'PENDING' COMMENT 'Forecast status: PENDING/GENERATED/CONFIRMED/ADJUSTED/ARCHIVED',
    forecast_model VARCHAR(64) COMMENT 'Forecast model used',
    confidence_level DECIMAL(5,2) COMMENT 'Confidence level',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_sku_code (sku_code),
    INDEX idx_warehouse_id (warehouse_id),
    INDEX idx_forecast_date (forecast_date),
    INDEX idx_status (status),
    INDEX idx_sku_forecast_date (sku_code, forecast_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Sales forecast data';

CREATE TABLE IF NOT EXISTS tb_forecast_model (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(128) COMMENT 'Model name',
    model_type VARCHAR(64) COMMENT 'Model type',
    model_version VARCHAR(32) COMMENT 'Model version',
    status VARCHAR(32) DEFAULT 'DRAFT' COMMENT 'Model status: DRAFT/TRAINING/TRAINED/DEPLOYED/DEPRECATED',
    accuracy DECIMAL(5,2) COMMENT 'Model accuracy',
    training_days INT COMMENT 'Training days',
    feature_config TEXT COMMENT 'Feature configuration JSON',
    algorithm VARCHAR(64) COMMENT 'Algorithm used',
    trained_at TIMESTAMP COMMENT 'Training completion time',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model_name (model_name),
    INDEX idx_model_type (model_type),
    INDEX idx_status (status),
    INDEX idx_model_type_status (model_type, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Forecast model configuration';
