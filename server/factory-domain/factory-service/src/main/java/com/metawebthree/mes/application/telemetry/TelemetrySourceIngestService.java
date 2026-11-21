package com.metawebthree.mes.application.telemetry;

import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.service.scada.ScadaDomainService;
import com.metawebthree.mes.domain.telemetry.TelemetryPoint;
import com.metawebthree.mes.domain.telemetry.TelemetrySource;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;

@Slf4j
@Service
public class TelemetrySourceIngestService {

    private final TelemetrySource source;
    private final ScadaDomainService scadaDomainService;

    public TelemetrySourceIngestService(TelemetrySource source, ScadaDomainService scadaDomainService) {
        this.source = source;
        this.scadaDomainService = scadaDomainService;
    }

    @PostConstruct
    public void start() {
        source.connect();
        source.subscribe(this::ingest);
        log.info("Telemetry source {} started", source.name());
    }

    @PreDestroy
    public void stop() {
        source.close();
    }

    void ingest(TelemetryPoint point) {
        TelemetryMetric metric = new TelemetryMetric();
        metric.setMetricCode(point.pointKey());
        metric.setMetricName(point.pointKey());
        metric.setValue(point.value());
        metric.setUnit(point.unit());
        metric.setQuality(point.quality());
        LocalDateTime collectTime = LocalDateTime.ofInstant(point.timestamp(), ZoneOffset.UTC);
        scadaDomainService.ingestMetrics(point.equipmentCode(), source.name(), List.of(metric), collectTime);
    }
}
