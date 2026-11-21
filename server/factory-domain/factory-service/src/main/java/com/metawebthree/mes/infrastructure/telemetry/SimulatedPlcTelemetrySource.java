package com.metawebthree.mes.infrastructure.telemetry;

import com.metawebthree.mes.domain.telemetry.TelemetryPoint;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import com.metawebthree.mes.domain.telemetry.TelemetrySourceException;
import lombok.extern.slf4j.Slf4j;

import java.time.Instant;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * A deterministic simulated PLC source for demos and tests. It produces
 * standard TelemetryPoints (quality + timestamp) through the same SPI as real
 * sources, so switching to a real OPC UA / S7 adapter only swaps one bean.
 * Never enabled in prod by default.
 */
@Slf4j
public class SimulatedPlcTelemetrySource implements TelemetrySource {

    private final String deviceCode;
    private final List<String> tags;
    private final long intervalMs;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private final List<Consumer<TelemetryPoint>> consumers = new CopyOnWriteArrayList<>();
    private long tick;

    public SimulatedPlcTelemetrySource(String deviceCode, List<String> tags, long intervalMs) {
        this.deviceCode = deviceCode;
        this.tags = List.copyOf(tags);
        this.intervalMs = intervalMs;
    }

    @Override
    public String name() {
        return "SIMULATED_PLC";
    }

    @Override
    public void connect() throws TelemetrySourceException {
        scheduler.scheduleAtFixedRate(this::emitTick, intervalMs, intervalMs, TimeUnit.MILLISECONDS);
        log.info("Simulated PLC source started: device={}, tags={}, interval={}ms", deviceCode, tags, intervalMs);
    }

    @Override
    public void subscribe(Consumer<TelemetryPoint> consumer) throws TelemetrySourceException {
        consumers.add(consumer);
    }

    @Override
    public List<TelemetryPoint> read() throws TelemetrySourceException {
        return List.of();
    }

    @Override
    public void close() {
        scheduler.shutdownNow();
    }

    private void emitTick() {
        Instant now = Instant.now();
        for (String tag : tags) {
            double value = Math.round(Math.sin((tick + tag.hashCode() % 100) / 10.0) * 1000.0) / 10.0;
            TelemetryPoint point = new TelemetryPoint(deviceCode, tag, value, null,
                TelemetryPoint.QUALITY_GOOD, now);
            consumers.forEach(consumer -> consumer.accept(point));
        }
        tick++;
    }
}
