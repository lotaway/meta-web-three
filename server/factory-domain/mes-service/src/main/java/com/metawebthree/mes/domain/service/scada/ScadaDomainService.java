package com.metawebthree.mes.domain.service.scada;

import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import java.time.LocalDateTime;
import java.util.List;

public interface ScadaDomainService {
    TelemetryRecord ingestTelemetry(String equipmentCode, String topic, String payload, LocalDateTime collectTime);
    TelemetryRecord ingestMetrics(String equipmentCode, String topic, List<TelemetryMetric> metrics, LocalDateTime collectTime);
    DeviceCommand dispatchCommand(String equipmentCode, DeviceCommand.CommandType commandType, String payload, String createdBy);
    void processCommandResponse(String commandCode, boolean success, String message);
    List<TelemetryRecord> getLatestTelemetry(String equipmentCode, int limit);
    TelemetryRecord getLatestTelemetryByMetric(String equipmentCode, String metricCode);
}
