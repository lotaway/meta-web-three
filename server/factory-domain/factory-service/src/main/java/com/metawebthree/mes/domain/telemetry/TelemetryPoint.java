package com.metawebthree.mes.domain.telemetry;

import java.time.Instant;

public record TelemetryPoint(
    String equipmentCode,
    String pointKey,
    Double value,
    String unit,
    String quality,
    Instant timestamp
) {
    public static final String QUALITY_GOOD = "GOOD";
    public static final String QUALITY_BAD = "BAD";

    public TelemetryPoint {
        quality = quality == null ? QUALITY_GOOD : quality;
        timestamp = timestamp == null ? Instant.now() : timestamp;
    }
}
