package com.metawebthree.mes.domain.telemetry;

public class TelemetrySourceException extends RuntimeException {

    public TelemetrySourceException(String message) {
        super(message);
    }

    public TelemetrySourceException(String message, Throwable cause) {
        super(message, cause);
    }
}
