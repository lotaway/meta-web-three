package com.metawebthree.mes.infrastructure.influx;

import com.influxdb.client.InfluxDBClient;
import com.influxdb.client.QueryApi;
import com.influxdb.client.WriteApiBlocking;
import com.influxdb.client.domain.WritePrecision;
import com.influxdb.client.write.Point;
import com.influxdb.query.FluxRecord;
import com.influxdb.query.FluxTable;
import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
public class InfluxTelemetryStore {

    private static final String MEASUREMENT = "telemetry";
    private static final String TAG_EQUIPMENT = "equipmentCode";

    private final InfluxDBClient client;
    private final String bucket;

    public InfluxTelemetryStore(InfluxDBClient client, String bucket) {
        this.client = client;
        this.bucket = bucket;
    }

    public void write(TelemetryRecord record) {
        Point point = Point.measurement(MEASUREMENT)
            .addTag(TAG_EQUIPMENT, record.getEquipmentCode())
            .time(toInstant(record.getCollectTime()), WritePrecision.MS);
        for (TelemetryMetric metric : record.getMetrics()) {
            if (metric.getValue() != null) {
                point.addField(metric.getMetricCode(), metric.getValue());
            }
        }
        WriteApiBlocking writeApi = client.getWriteApiBlocking();
        writeApi.writePoint(point);
    }

    public List<TelemetryRecord> queryRange(String equipmentCode, LocalDateTime start, LocalDateTime end) {
        String flux = String.format(
            "from(bucket: \"%s\") |> range(start: %s, stop: %s) " +
            "|> filter(fn: (r) => r._measurement == \"%s\" and r.%s == \"%s\")",
            bucket, toInstant(start), toInstant(end), MEASUREMENT, TAG_EQUIPMENT, equipmentCode);
        QueryApi queryApi = client.getQueryApi();
        List<FluxTable> tables = queryApi.query(flux);
        return toRecords(tables, equipmentCode);
    }

    private List<TelemetryRecord> toRecords(List<FluxTable> tables, String equipmentCode) {
        Map<String, List<FluxRecord>> byTime = new java.util.LinkedHashMap<>();
        for (FluxTable table : tables) {
            for (FluxRecord record : table.getRecords()) {
                String key = String.valueOf(record.getTime());
                byTime.computeIfAbsent(key, (k) -> new ArrayList<>()).add(record);
            }
        }
        List<TelemetryRecord> result = new ArrayList<>();
        for (List<FluxRecord> group : byTime.values()) {
            result.add(toRecord(group, equipmentCode));
        }
        return result;
    }

    private TelemetryRecord toRecord(List<FluxRecord> group, String equipmentCode) {
        FluxRecord first = group.get(0);
        LocalDateTime collectTime = LocalDateTime.ofInstant(first.getTime(), ZoneOffset.UTC);
        TelemetryRecord record = new TelemetryRecord();
        record.create(equipmentCode, MEASUREMENT, collectTime);
        for (FluxRecord fluxRecord : group) {
            String field = fluxRecord.getField();
            if (field != null && fluxRecord.getValue() instanceof Number) {
                TelemetryMetric metric = new TelemetryMetric();
                metric.setMetricCode(field);
                metric.setMetricName(field);
                metric.setValue(((Number) fluxRecord.getValue()).doubleValue());
                record.getMetrics().add(metric);
            }
        }
        return record;
    }

    private Instant toInstant(LocalDateTime time) {
        return (time != null ? time : LocalDateTime.now()).toInstant(ZoneOffset.UTC);
    }
}
