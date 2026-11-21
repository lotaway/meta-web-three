package com.metawebthree.mes.infrastructure.telemetry;

import com.metawebthree.mes.domain.telemetry.TelemetryPoint;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SimulatedPlcTelemetrySourceTest {

    @Test
    void producesStandardPointsThroughTheSpi() throws Exception {
        SimulatedPlcTelemetrySource source = new SimulatedPlcTelemetrySource(
            "SIM-PLC-001", List.of("temperature", "vibration"), 20);
        List<TelemetryPoint> received = new ArrayList<>();
        CountDownLatch latch = new CountDownLatch(2);
        source.subscribe(point -> {
            received.add(point);
            latch.countDown();
        });

        assertEquals("SIMULATED_PLC", source.name());
        source.connect();
        try {
            assertTrue(latch.await(3, TimeUnit.SECONDS));
        } finally {
            source.close();
        }

        assertEquals(2, received.size());
        TelemetryPoint first = received.get(0);
        assertEquals("SIM-PLC-001", first.equipmentCode());
        assertTrue(List.of("temperature", "vibration").contains(first.pointKey()));
        assertEquals(TelemetryPoint.QUALITY_GOOD, first.quality());
        assertNotNull(first.timestamp());
    }
}
