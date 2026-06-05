package com.metawebthree.mes.domain.entity.scada;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class TelemetryRecord {
    private Long id;
    private String equipmentCode;
    private String topic;
    private LocalDateTime collectTime;
    private List<TelemetryMetric> metrics;
    private LocalDateTime createdAt;

    public enum DataSource {
        MQTT, API, MANUAL, OPC_UA, MODBUS
    }

    public void create(String equipmentCode, String topic, LocalDateTime collectTime) {
        this.equipmentCode = equipmentCode;
        this.topic = topic;
        this.collectTime = collectTime != null ? collectTime : LocalDateTime.now();
        this.metrics = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
    }

    public void addMetric(String metricCode, String metricName, Double value, String unit) {
        TelemetryMetric metric = new TelemetryMetric();
        metric.setMetricCode(metricCode);
        metric.setMetricName(metricName);
        metric.setValue(value);
        metric.setUnit(unit);
        this.metrics.add(metric);
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEquipmentCode() { return equipmentCode; }
    public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
    public String getTopic() { return topic; }
    public void setTopic(String topic) { this.topic = topic; }
    public LocalDateTime getCollectTime() { return collectTime; }
    public void setCollectTime(LocalDateTime collectTime) { this.collectTime = collectTime; }
    public List<TelemetryMetric> getMetrics() { return metrics; }
    public void setMetrics(List<TelemetryMetric> metrics) { this.metrics = metrics; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
