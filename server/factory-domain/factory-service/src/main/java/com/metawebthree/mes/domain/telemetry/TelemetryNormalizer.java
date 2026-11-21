package com.metawebthree.mes.domain.telemetry;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class TelemetryNormalizer {

    private final TelemetryMappingProperties properties;

    public TelemetryNormalizer(TelemetryMappingProperties properties) {
        this.properties = properties;
    }

    public List<TelemetryPoint> normalize(String topic, Map<String, Object> payload) {
        String equipmentCode = extractEquipmentCode(topic);
        if (equipmentCode == null) return List.of();
        Instant timestamp = extractTimestamp(payload);
        String quality = extractQuality(payload);
        List<TelemetryPoint> points = new ArrayList<>();
        for (Map.Entry<String, Object> entry : payload.entrySet()) {
            if (isMetaField(entry.getKey())) continue;
            if (!(entry.getValue() instanceof Number)) continue;
            points.add(new TelemetryPoint(
                equipmentCode,
                aliasOf(entry.getKey()),
                ((Number) entry.getValue()).doubleValue(),
                extractUnit(payload, entry.getKey()),
                quality,
                timestamp));
        }
        return points;
    }

    private boolean isMetaField(String key) {
        return key.equals(properties.getTimestampField())
            || key.equals(properties.getQualityField());
    }

    private String aliasOf(String key) {
        return properties.getPointAliases().getOrDefault(key, key);
    }

    private Instant extractTimestamp(Map<String, Object> payload) {
        Object raw = payload.get(properties.getTimestampField());
        if (raw instanceof Number) return Instant.ofEpochMilli(((Number) raw).longValue());
        if (raw instanceof String) {
            try { return Instant.parse((String) raw); } catch (Exception ignored) { }
        }
        return Instant.now();
    }

    private String extractQuality(Map<String, Object> payload) {
        Object raw = payload.get(properties.getQualityField());
        if (raw == null) return TelemetryPoint.QUALITY_GOOD;
        String key = String.valueOf(raw).toLowerCase();
        return properties.getQualityValues().getOrDefault(key, TelemetryPoint.QUALITY_GOOD);
    }

    private String extractUnit(Map<String, Object> payload, String pointKey) {
        Object units = payload.get(properties.getUnitField());
        if (units instanceof Map) {
            Object unit = ((Map<?, ?>) units).get(pointKey);
            return unit == null ? null : String.valueOf(unit);
        }
        return units == null ? null : String.valueOf(units);
    }

    private String extractEquipmentCode(String topic) {
        if (topic == null || topic.isBlank()) return null;
        String[] segments = topic.split("/");
        String segment = properties.getDeviceCodeSegment();
        if (segment != null && segment.matches("\\d+")) {
            int idx = Integer.parseInt(segment);
            return idx >= 0 && idx < segments.length ? segments[idx] : null;
        }
        if (segment != null && !segment.isBlank()) {
            for (String s : segments) {
                if (segment.equalsIgnoreCase(s)) continue;
                int idx = indexOfSegment(segments, segment);
                if (idx >= 0 && idx + 1 < segments.length) return segments[idx + 1];
            }
        }
        return segments.length >= 3 ? segments[segments.length - 2] : null;
    }

    private int indexOfSegment(String[] segments, String target) {
        for (int i = 0; i < segments.length; i++) {
            if (target.equalsIgnoreCase(segments[i])) return i;
        }
        return -1;
    }
}
