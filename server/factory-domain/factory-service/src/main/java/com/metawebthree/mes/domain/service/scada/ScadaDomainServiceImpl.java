package com.metawebthree.mes.domain.service.scada;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.AndonEvent;
import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.TelemetryMetric;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import com.metawebthree.mes.domain.repository.AndonEventRepository;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import com.metawebthree.mes.domain.repository.scada.DeviceCommandRepository;
import com.metawebthree.mes.domain.repository.scada.TelemetryRecordRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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
    private final EquipmentRepository equipmentRepository;
    private final AndonEventRepository andonEventRepository;
    private final WorkOrderRepository workOrderRepository;
    private final ObjectMapper objectMapper;

    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public ScadaDomainServiceImpl(TelemetryRecordRepository telemetryRecordRepository,
                                   DeviceCommandRepository deviceCommandRepository,
                                   EquipmentRepository equipmentRepository,
                                   AndonEventRepository andonEventRepository,
                                   WorkOrderRepository workOrderRepository,
                                   ObjectMapper objectMapper) {
        this.telemetryRecordRepository = telemetryRecordRepository;
        this.deviceCommandRepository = deviceCommandRepository;
        this.equipmentRepository = equipmentRepository;
        this.andonEventRepository = andonEventRepository;
        this.workOrderRepository = workOrderRepository;
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
            log.warn("Failed to parse telemetry payload for {}: {}", equipmentCode, e);
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

    @Override
    public DashboardMetrics getDashboardMetrics(String workshopId) {
        List<Equipment> equipments = workshopId != null && !workshopId.isEmpty()
            ? equipmentRepository.findByWorkshopId(workshopId)
            : equipmentRepository.findByStatusCode("ONLINE");

        int totalEquipment = equipments.size();
        int onlineEquipment = (int) equipments.stream()
            .filter(e -> e.getStatus() != null && e.getStatus() == Equipment.EquipmentStatus.ONLINE)
            .count();
        int runningEquipment = (int) equipments.stream()
            .filter(e -> e.getStatus() != null && e.getStatus() == Equipment.EquipmentStatus.RUNNING)
            .count();
        int idleEquipment = (int) equipments.stream()
            .filter(e -> e.getStatus() != null && e.getStatus() == Equipment.EquipmentStatus.IDLE)
            .count();
        int warningEquipment = (int) equipments.stream()
            .filter(e -> e.getStatus() != null && e.getStatus() == Equipment.EquipmentStatus.WARNING)
            .count();
        int errorEquipment = (int) equipments.stream()
            .filter(e -> e.getStatus() != null && e.getStatus() == Equipment.EquipmentStatus.ERROR)
            .count();
        int offlineEquipment = (int) equipments.stream()
            .filter(e -> e.getStatus() != null && e.getStatus() == Equipment.EquipmentStatus.OFFLINE)
            .count();

        double avgOee = equipments.stream()
            .filter(e -> e.getUtilizationRate() != null)
            .mapToDouble(Equipment::getUtilizationRate)
            .average()
            .orElse(0.0);

        int todayOutput = equipments.stream()
            .filter(e -> e.getTodayOutput() != null)
            .mapToInt(Equipment::getTodayOutput)
            .sum();

        List<AndonEvent> activeAlerts = andonEventRepository.findByStatus(AndonEvent.AndonEventStatus.PENDING);
        int activeAlertCount = activeAlerts.size();

        List<WorkOrder> pendingOrders = workOrderRepository.findByStatus(WorkOrder.WorkOrderStatus.PENDING);
        int pendingOrderCount = pendingOrders.size();

        return new DashboardMetrics(
            totalEquipment, onlineEquipment, runningEquipment, idleEquipment,
            warningEquipment, errorEquipment, offlineEquipment,
            Math.round(avgOee * 100.0) / 100.0, todayOutput, activeAlertCount, pendingOrderCount
        );
    }

    @Override
    public List<EquipmentStatusSummary> getEquipmentStatusSummary(String workshopId) {
        List<Equipment> equipments = workshopId != null && !workshopId.isEmpty()
            ? equipmentRepository.findByWorkshopId(workshopId)
            : equipmentRepository.findByStatusCode("ONLINE");

        return equipments.stream()
            .map(e -> new EquipmentStatusSummary(
                e.getId(),
                e.getEquipmentCode(),
                e.getEquipmentName(),
                e.getStatus() != null ? e.getStatus().name() : "UNKNOWN",
                e.getUtilizationRate(),
                e.getTodayOutput(),
                e.getLastHeartbeat() != null ? e.getLastHeartbeat().format(FORMATTER) : "-",
                e.getCurrentTaskNo()
            ))
            .collect(Collectors.toList());
    }

    @Override
    public List<AlertSummary> getActiveAlerts(String workshopId) {
        List<AndonEvent> events = andonEventRepository.findByStatus(AndonEvent.AndonEventStatus.PENDING);
        
        if (workshopId != null && !workshopId.isEmpty()) {
            events = events.stream()
                .filter(e -> workshopId.equals(e.getWorkshopId()))
                .collect(Collectors.toList());
        }

        return events.stream()
            .map(e -> {
                String eqName = null;
                if (e.getEquipmentId() != null) {
                    eqName = equipmentRepository.findById(Long.parseLong(e.getEquipmentId()))
                        .map(Equipment::getEquipmentName)
                        .orElse(null);
                }
                return new AlertSummary(
                    e.getId(),
                    e.getEventNo(),
                    e.getLevelCode(),
                    e.getLevelName(),
                    e.getEquipmentId(),
                    e.getEquipmentId(),
                    eqName,
                    e.getDescription(),
                    e.getStatus() != null ? e.getStatus().name() : "UNKNOWN",
                    e.getOccurredAt() != null ? e.getOccurredAt().format(FORMATTER) : "-",
                    e.getReporterName()
                );
            })
            .collect(Collectors.toList());
    }

    @Override
    public ProductionStats getProductionStats(String workshopId) {
        List<WorkOrder> orders = workshopId != null && !workshopId.isEmpty()
            ? workOrderRepository.findByWorkshopId(workshopId)
            : workOrderRepository.findAll();

        int totalWorkOrders = orders.size();
        int inProgressWorkOrders = (int) orders.stream()
            .filter(o -> o.getStatus() == WorkOrder.WorkOrderStatus.IN_PROGRESS)
            .count();
        int completedWorkOrders = (int) orders.stream()
            .filter(o -> o.getStatus() == WorkOrder.WorkOrderStatus.COMPLETED)
            .count();
        int pendingWorkOrders = (int) orders.stream()
            .filter(o -> o.getStatus() == WorkOrder.WorkOrderStatus.PENDING)
            .count();

        int todayOutput = orders.stream()
            .filter(o -> o.getCompletedQuantity() != null)
            .mapToInt(WorkOrder::getCompletedQuantity)
            .sum();

        int todayPlannedOutput = orders.stream()
            .filter(o -> o.getQuantity() != null)
            .mapToInt(WorkOrder::getQuantity)
            .sum();

        return new ProductionStats(
            totalWorkOrders, inProgressWorkOrders, completedWorkOrders, pendingWorkOrders,
            todayOutput, todayPlannedOutput, 0, 0,
            todayPlannedOutput > 0 ? Math.round((double) todayOutput / todayPlannedOutput * 10000.0) / 100.0 : 0.0
        );
    }
}
