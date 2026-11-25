-- SCADA telemetry store. Backs ScadaTelemetryRecordDO / mes_scada_telemetry.
-- Without this table, telemetry ingest fails at runtime even when MQTT is enabled.
CREATE TABLE IF NOT EXISTS mes_scada_telemetry (
    id BIGSERIAL PRIMARY KEY,
    equipment_code VARCHAR(100) NOT NULL,
    topic VARCHAR(255),
    payload TEXT,
    collect_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scada_telemetry_equipment_time
    ON mes_scada_telemetry (equipment_code, collect_time DESC);
