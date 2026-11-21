package com.metawebthree.mes.domain.telemetry;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TelemetryNormalizerTest {

    private final TelemetryNormalizer normalizer = new TelemetryNormalizer(new TelemetryMappingProperties());

    @Test
    void normalizesStandardMqttPayload() {
        List<TelemetryPoint> points = normalizer.normalize(
            "mes/equipment/AGV-001/telemetry",
            Map.of("temperature", 23.5, "timestamp", 1700000000000L, "quality", "good"));

        assertEquals(1, points.size());
        TelemetryPoint point = points.get(0);
        assertEquals("AGV-001", point.equipmentCode());
        assertEquals("temperature", point.pointKey());
        assertEquals(23.5, point.value());
        assertEquals(TelemetryPoint.QUALITY_GOOD, point.quality());
        assertEquals(Instant.ofEpochMilli(1700000000000L), point.timestamp());
    }

    @Test
    void mapsQualityAndIgnoresMetaFields() {
        List<TelemetryPoint> points = normalizer.normalize(
            "mes/equipment/PLC-01/telemetry",
            Map.of("vibration", 1.2, "quality", "bad", "timestamp", 1700000000000L));

        assertEquals(1, points.size());
        assertEquals(TelemetryPoint.QUALITY_BAD, points.get(0).quality());
    }

    @Test
    void appliesPointAliasAndUnitMap() {
        TelemetryMappingProperties props = new TelemetryMappingProperties();
        props.getPointAliases().put("t", "temperature");
        props.getUnitField();
        TelemetryNormalizer custom = new TelemetryNormalizer(props);

        List<TelemetryPoint> points = custom.normalize(
            "mes/equipment/DEV/telemetry", Map.of("t", 42.0));

        assertEquals("temperature", points.get(0).pointKey());
    }

    @Test
    void extractsDeviceFromIsa95UnsTopic() {
        TelemetryMappingProperties props = new TelemetryMappingProperties();
        props.setTopicPatterns(List.of("+/+/+/+/+/+/+"));
        props.setDeviceCodeSegment("Cell");
        List<TelemetryPoint> points = new TelemetryNormalizer(props).normalize(
            "Enterprise/Site/Area/Line/Cell/ROBOT-02/Tag/temperature",
            Map.of("value", 7.0));

        assertEquals("ROBOT-02", points.get(0).equipmentCode());
    }

    @Test
    void returnsEmptyForUnexpectedTopic() {
        assertTrue(normalizer.normalize(null, Map.of("v", 1.0)).isEmpty());
    }
}
