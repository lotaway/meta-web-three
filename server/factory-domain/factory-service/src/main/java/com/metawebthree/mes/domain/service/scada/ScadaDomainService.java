package com.metawebthree.mes.domain.service.scada;

import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public interface ScadaDomainService {
    TelemetryRecord ingestTelemetry(String equipmentCode, String topic, String payload, LocalDateTime collectTime);
    TelemetryRecord ingestMetrics(String equipmentCode, String topic, List<TelemetryMetric> metrics, LocalDateTime collectTime);
    DeviceCommand dispatchCommand(String equipmentCode, DeviceCommand.CommandType commandType, String payload, String createdBy);
    void processCommandResponse(String commandCode, boolean success, String message);
    List<TelemetryRecord> getLatestTelemetry(String equipmentCode, int limit);
    TelemetryRecord getLatestTelemetryByMetric(String equipmentCode, String metricCode);
    DashboardMetrics getDashboardMetrics(String workshopId);
    List<EquipmentStatusSummary> getEquipmentStatusSummary(String workshopId);
    List<AlertSummary> getActiveAlerts(String workshopId);
    ProductionStats getProductionStats(String workshopId);

    record DashboardMetrics(
        int totalEquipment,
        int onlineEquipment,
        int runningEquipment,
        int idleEquipment,
        int warningEquipment,
        int errorEquipment,
        int offlineEquipment,
        double avgOee,
        int todayOutput,
        int activeAlerts,
        int pendingWorkOrders
    ) {}

    record EquipmentStatusSummary(
        Long equipmentId,
        String equipmentCode,
        String equipmentName,
        String status,
        Double oee,
        Integer todayOutput,
        String lastHeartbeat,
        String currentTaskNo
    ) {}

    record AlertSummary(
        Long id,
        String eventNo,
        String levelCode,
        String levelName,
        String equipmentId,
        String equipmentCode,
        String equipmentName,
        String description,
        String status,
        String occurredAt,
        String reporterName
    ) {}

    record ProductionStats(
        int totalWorkOrders,
        int inProgressWorkOrders,
        int completedWorkOrders,
        int pendingWorkOrders,
        int todayOutput,
        int todayPlannedOutput,
        int totalTasks,
        int completedTasks,
        double completionRate
    ) {}
}
