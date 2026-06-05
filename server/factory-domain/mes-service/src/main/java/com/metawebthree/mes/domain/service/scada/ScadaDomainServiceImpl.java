package com.metawebthree.mes.domain.service.scada;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import com.metawebthree.mes.domain.repository.scada.DeviceCommandRepository;
import com.metawebthree.mes.domain.repository.scada.TelemetryRecordRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@Slf4j
@Service
public class ScadaDomainServiceImpl implements ScadaDomainService {

    private final TelemetryRecordRepository telemetryRecordRepository;
    private final DeviceCommandRepository deviceCommandRepository;
    private final ObjectMapper objectMapper;

    public ScadaDomainServiceImpl(TelemetryRecordRepository telemetryRecordRepository,
                                   DeviceCommandRepository deviceCommandRepository,
                                   ObjectMapper objectMapper) {
        this.telemetryRecordRepository = telemetryRecordRepository;
        this.deviceCommandRepository = deviceCommandRepository;
        this.objectMapper = objectMapper;
    }

    @Override
    public TelemetryRecord ingestTelemetry(String equipmentCode, String topic, String payload, LocalDateTime collectTime) {
        TelemetryRecord record = new TelemetryRecord();
        record.create(equipmentCode, topic, collectTime != null ? collectTime : LocalDateTime.now());

        try {
            Map<String, Object> data = objectMapper.readValue(payload, new TypeReference<Map<String, Object>>() {});
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                TelemetryMetric metric = new TelemetryMetric();
                metric.setMetricCode(entry.getKey());
                metric.setMetricName(entry.getKey());
                if (entry.getValue() instanceof Number) {
                    metric.setValue(((Number) entry.getValue()).doubleValue());
                }
                record.getMetrics().add(metric);
            }
        } catch (Exception e) {
            log.warn("Failed to parse telemetry payload for {}: {}", equipmentCode, e.getMessage());
        }
        return telemetryRecordRepository.save(record);
    }

    @Override
    public TelemetryRecord ingestMetrics(String equipmentCode, String topic,
                                          List<TelemetryMetric> metrics, LocalDateTime collectTime) {
        TelemetryRecord record = new TelemetryRecord();
        record.create(equipmentCode, topic, collectTime != null ? collectTime : LocalDateTime.now());
        if (metrics != null) record.setMetrics(metrics);
        return telemetryRecordRepository.save(record);
    }

    @Override
    public DeviceCommand dispatchCommand(String equipmentCode, DeviceCommand.CommandType commandType,
                                          String payload, String createdBy) {
        String commandCode = "CMD-" + System.currentTimeMillis() + "-" + UUID.randomUUID().toString().substring(0, 8);
        DeviceCommand command = new DeviceCommand();
        command.create(commandCode, equipmentCode, commandType, payload, createdBy);
        command.markSent();
        DeviceCommand saved = deviceCommandRepository.save(command);
        log.info("Dispatched command {} to {}: {}", commandCode, equipmentCode, commandType);
        return saved;
    }

    @Override
    public void processCommandResponse(String commandCode, boolean success, String message) {
        deviceCommandRepository.findByCommandCode(commandCode).ifPresent(cmd -> {
            if (success) cmd.markExecuted(message);
            else cmd.markFailed(message);
            deviceCommandRepository.update(cmd);
        });
    }

    @Override
    public List<TelemetryRecord> getLatestTelemetry(String equipmentCode, int limit) {
        List<TelemetryRecord> all = telemetryRecordRepository.findByEquipmentCode(equipmentCode);
        Collections.reverse(all);
        return all.stream().limit(limit).collect(Collectors.toList());
    }

    @Override
    public TelemetryRecord getLatestTelemetryByMetric(String equipmentCode, String metricCode) {
        List<TelemetryRecord> records = telemetryRecordRepository.findByEquipmentCode(equipmentCode);
        return records.stream()
            .filter(r -> r.getMetrics() != null && r.getMetrics().stream()
                .anyMatch(m -> metricCode.equals(m.getMetricCode())))
            .reduce((first, second) -> second)
            .orElse(null);
    }
}
