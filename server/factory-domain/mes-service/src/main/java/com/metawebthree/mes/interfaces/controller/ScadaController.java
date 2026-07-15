package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.MesPermissions;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand.CommandType;
import com.metawebthree.mes.domain.entity.scada.TelemetryRecord;
import com.metawebthree.mes.domain.repository.scada.DeviceCommandRepository;
import com.metawebthree.mes.domain.repository.scada.TelemetryRecordRepository;
import com.metawebthree.mes.domain.service.scada.ScadaDomainService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/mes/scada")
public class ScadaController {

    private final ScadaDomainService scadaDomainService;
    private final TelemetryRecordRepository telemetryRecordRepository;
    private final DeviceCommandRepository deviceCommandRepository;

    public ScadaController(ScadaDomainService scadaDomainService,
            TelemetryRecordRepository telemetryRecordRepository,
            DeviceCommandRepository deviceCommandRepository) {
        this.scadaDomainService = scadaDomainService;
        this.telemetryRecordRepository = telemetryRecordRepository;
        this.deviceCommandRepository = deviceCommandRepository;
    }

    @PostMapping("/telemetry/ingest")
    @RequirePermission(MesPermissions.SCADA_TELEMETRY_READ)
    public ResponseEntity<TelemetryRecord> ingestTelemetry(@RequestBody IngestTelemetryRequest req) {
        return ResponseEntity.ok(scadaDomainService.ingestTelemetry(
                req.getEquipmentCode(), req.getTopic(), req.getPayload(), req.getCollectTime()));
    }

    @GetMapping("/telemetry/{equipmentCode}")
    @RequirePermission(MesPermissions.SCADA_TELEMETRY_READ)
    public ResponseEntity<List<TelemetryRecord>> getTelemetry(
            @PathVariable String equipmentCode,
            @RequestParam(defaultValue = "10") int limit) {
        return ResponseEntity.ok(scadaDomainService.getLatestTelemetry(equipmentCode, limit));
    }

    @GetMapping("/telemetry/{equipmentCode}/range")
    @RequirePermission(MesPermissions.SCADA_TELEMETRY_READ)
    public ResponseEntity<List<TelemetryRecord>> getTelemetryByRange(
            @PathVariable String equipmentCode,
            @RequestParam String start,
            @RequestParam String end) {
        return ResponseEntity.ok(telemetryRecordRepository.findByEquipmentCodeAndTimeRange(
                equipmentCode, LocalDateTime.parse(start), LocalDateTime.parse(end)));
    }

    @PostMapping("/commands")
    @RequirePermission(MesPermissions.SCADA_COMMAND_DISPATCH)
    public ResponseEntity<DeviceCommand> dispatchCommand(@RequestBody DispatchCommandRequest req) {
        return ResponseEntity.ok(scadaDomainService.dispatchCommand(
                req.getEquipmentCode(), req.getCommandType(), req.getPayload(), req.getCreatedBy()));
    }

    @GetMapping("/commands/{equipmentCode}")
    @RequirePermission(MesPermissions.SCADA_COMMAND_READ)
    public ResponseEntity<List<DeviceCommand>> getCommands(
            @PathVariable String equipmentCode,
            @RequestParam(required = false) String status) {
        if (status != null) {
            return ResponseEntity.ok(deviceCommandRepository.findByEquipmentCodeAndStatus(
                    equipmentCode, DeviceCommand.CommandStatus.valueOf(status)));
        }
        return ResponseEntity.ok(deviceCommandRepository.findByEquipmentCode(equipmentCode));
    }

    @GetMapping("/commands/status/{commandCode}")
    @RequirePermission(MesPermissions.SCADA_COMMAND_READ)
    public ResponseEntity<DeviceCommand> getCommandStatus(@PathVariable String commandCode) {
        return deviceCommandRepository.findByCommandCode(commandCode)
                .map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/dashboard/metrics")
    @RequirePermission(MesPermissions.SCADA_READ)
    public ResponseEntity<ScadaDomainService.DashboardMetrics> getDashboardMetrics(
            @RequestParam(required = false) String workshopId) {
        return ResponseEntity.ok(scadaDomainService.getDashboardMetrics(workshopId));
    }

    @GetMapping("/dashboard/equipment")
    @RequirePermission(MesPermissions.SCADA_READ)
    public ResponseEntity<List<ScadaDomainService.EquipmentStatusSummary>> getEquipmentStatusSummary(
            @RequestParam(required = false) String workshopId) {
        return ResponseEntity.ok(scadaDomainService.getEquipmentStatusSummary(workshopId));
    }

    @GetMapping("/dashboard/alerts")
    @RequirePermission(MesPermissions.SCADA_READ)
    public ResponseEntity<List<ScadaDomainService.AlertSummary>> getActiveAlerts(
            @RequestParam(required = false) String workshopId) {
        return ResponseEntity.ok(scadaDomainService.getActiveAlerts(workshopId));
    }

    @GetMapping("/dashboard/production")
    @RequirePermission(MesPermissions.SCADA_READ)
    public ResponseEntity<ScadaDomainService.ProductionStats> getProductionStats(
            @RequestParam(required = false) String workshopId) {
        return ResponseEntity.ok(scadaDomainService.getProductionStats(workshopId));
    }

    public static class IngestTelemetryRequest {
        private String equipmentCode;
        private String topic;
        private String payload;
        private LocalDateTime collectTime;

        public String getEquipmentCode() {
            return equipmentCode;
        }

        public void setEquipmentCode(String equipmentCode) {
            this.equipmentCode = equipmentCode;
        }

        public String getTopic() {
            return topic;
        }

        public void setTopic(String topic) {
            this.topic = topic;
        }

        public String getPayload() {
            return payload;
        }

        public void setPayload(String payload) {
            this.payload = payload;
        }

        public LocalDateTime getCollectTime() {
            return collectTime;
        }

        public void setCollectTime(LocalDateTime collectTime) {
            this.collectTime = collectTime;
        }
    }

    public static class DispatchCommandRequest {
        private String equipmentCode;
        private CommandType commandType;
        private String payload;
        private String createdBy;

        public String getEquipmentCode() {
            return equipmentCode;
        }

        public void setEquipmentCode(String equipmentCode) {
            this.equipmentCode = equipmentCode;
        }

        public CommandType getCommandType() {
            return commandType;
        }

        public void setCommandType(CommandType commandType) {
            this.commandType = commandType;
        }

        public String getPayload() {
            return payload;
        }

        public void setPayload(String payload) {
            this.payload = payload;
        }

        public String getCreatedBy() {
            return createdBy;
        }

        public void setCreatedBy(String createdBy) {
            this.createdBy = createdBy;
        }
    }
}
