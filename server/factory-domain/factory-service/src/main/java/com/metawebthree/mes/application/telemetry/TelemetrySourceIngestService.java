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

    private final List<TelemetrySource> sources;
    private final ScadaDomainService scadaDomainService;

    public TelemetrySourceIngestService(List<TelemetrySource> sources, ScadaDomainService scadaDomainService) {
        this.sources = sources;
        this.scadaDomainService = scadaDomainService;
    }

    @PostConstruct
    public void start() {
        for (TelemetrySource source : sources) {
            try {
                source.subscribe(point -> ingest(source.name(), point));
                source.connect();
                log.info("Telemetry source {} started", source.name());
            } catch (Exception e) {
                log.error("Failed to start telemetry source {}: {}", source.name(), e.getMessage());
            }
        }
    }

    @PreDestroy
    public void stop() {
        sources.forEach(TelemetrySource::close);
    }

    void ingest(String sourceName, TelemetryPoint point) {
        TelemetryMetric metric = new TelemetryMetric();
        metric.setMetricCode(point.pointKey());
        metric.setMetricName(point.pointKey());
        metric.setValue(point.value());
        metric.setUnit(point.unit());
        metric.setQuality(point.quality());
        LocalDateTime collectTime = LocalDateTime.ofInstant(point.timestamp(), ZoneOffset.UTC);
        scadaDomainService.ingestMetrics(point.equipmentCode(), sourceName, List.of(metric), collectTime);
    }
}
