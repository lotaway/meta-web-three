package com.metawebthree.digitaltwin.infrastructure.event;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public final class DigitalTwinEventPayload {

    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE_TIME;

    private DigitalTwinEventPayload() {
    }

    public static Map<String, Object> of(DigitalTwinEventType type, Map<String, Object> fields) {
        Map<String, Object> data = new HashMap<>(fields);
        data.put("eventId", UUID.randomUUID().toString());
        data.put("eventType", type.name());
        data.put("timestamp", System.currentTimeMillis());
        return data;
    }

    public static String format(LocalDateTime time) {
        return time == null ? null : time.format(FORMATTER);
    }
}
