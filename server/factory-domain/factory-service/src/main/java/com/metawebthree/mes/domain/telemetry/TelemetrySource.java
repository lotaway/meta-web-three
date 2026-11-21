package com.metawebthree.mes.domain.telemetry;

import java.util.List;
import java.util.function.Consumer;

public interface TelemetrySource {

    String name();

    void connect() throws TelemetrySourceException;

    void subscribe(Consumer<TelemetryPoint> consumer) throws TelemetrySourceException;

    List<TelemetryPoint> read() throws TelemetrySourceException;

    void close();
}
