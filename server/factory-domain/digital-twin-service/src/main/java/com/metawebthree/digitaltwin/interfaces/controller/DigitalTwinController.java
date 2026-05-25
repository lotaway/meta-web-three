package com.metawebthree.digitaltwin.interfaces.controller;

import com.metawebthree.digitaltwin.application.command.DigitalTwinCommandService;
import com.metawebthree.digitaltwin.application.query.DigitalTwinQueryService;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.digitaltwin.interfaces.dto.*;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/digital-twin")
@CrossOrigin(origins = {"http://localhost:5173", "http://127.0.0.1:5173"})
public class DigitalTwinController {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinController.class);

    private final DigitalTwinCommandService commandService;
    private final DigitalTwinQueryService queryService;

    public DigitalTwinController(
            DigitalTwinCommandService commandService,
            DigitalTwinQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // Device endpoints
    @RequirePermission("dt:device:create")
    @PostMapping("/device")
    public ResponseEntity<Map<String, Object>> registerDevice(
            @Valid @RequestBody RegisterDeviceRequest request,
            @RequestHeader(value = "X-User-Id", required = false) String userId,
            @RequestHeader(value = "X-User-Role", required = false) String userRole) {
        logger.debug("Register device by user: {}, role: {}", userId, userRole);
        Long id = commandService.registerDevice(
            request.getDeviceCode(),
            request.getDeviceName(),
            request.getDeviceType(),
            request.getWorkshopId(),
            request.getProductionLineId()
        );
        return ResponseEntity.ok(Map.of("deviceId", id));
    }

    @RequirePermission("dt:device:update")
    @PostMapping("/device/{deviceCode}/status")
    public ResponseEntity<Void> updateDeviceStatus(
            @PathVariable String deviceCode,
            @Valid @RequestBody UpdateDeviceStatusRequest request,
            @RequestHeader(value = "X-User-Id", required = false) String userId,
            @RequestHeader(value = "X-User-Role", required = false) String userRole) {
        logger.debug("Update device status by user: {}, role: {}", userId, userRole);
        Device.DeviceStatus deviceStatus;
        try {
            deviceStatus = Device.DeviceStatus.valueOf(request.getStatus().toUpperCase());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }
        commandService.updateDeviceStatus(deviceCode, deviceStatus);
        return ResponseEntity.ok().build();
    }

    @RequirePermission("dt:device:update")
    @PostMapping("/device/{deviceCode}/heartbeat")
    public ResponseEntity<Void> deviceHeartbeat(
            @PathVariable String deviceCode,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        commandService.deviceHeartbeat(deviceCode);
        return ResponseEntity.ok().build();
    }

    @RequirePermission("dt:device:update")
    @PostMapping("/device/{deviceCode}/position")
    public ResponseEntity<Void> updateDevicePosition(
            @PathVariable String deviceCode,
            @Valid @RequestBody UpdateDevicePositionRequest request,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        Double x = request.getX();
        Double y = request.getY();
        Double z = request.getZ() != null ? request.getZ() : 0.0;
        Double rotation = 0.0;
        commandService.updateDevicePosition(deviceCode, x, y, z, rotation);
        return ResponseEntity.ok().build();
    }

    @RequirePermission("dt:device:read")
    @GetMapping("/device/{id}")
    public ResponseEntity<?> getDevice(@PathVariable Long id) {
        return queryService.getDeviceById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @RequirePermission("dt:device:read")
    @GetMapping("/devices")
    public ResponseEntity<?> getAllDevices(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        // Clamp page size to prevent abuse
        int limitedSize = Math.min(size, 100);
        return ResponseEntity.ok(queryService.getDevicesPaginated(page, limitedSize));
    }

    @RequirePermission("dt:device:read")
    @GetMapping("/workshop/{workshopId}/devices")
    public ResponseEntity<?> getWorkshopDevices(@PathVariable String workshopId) {
        return ResponseEntity.ok(queryService.getWorkshopDevices(workshopId));
    }

    @RequirePermission("dt:device:read")
    @GetMapping("/devices/online")
    public ResponseEntity<?> getOnlineDevices() {
        return ResponseEntity.ok(queryService.getOnlineDevices());
    }

    // Workshop endpoints
    @RequirePermission("dt:workshop:create")
    @PostMapping("/workshop")
    public ResponseEntity<Map<String, Object>> createWorkshop(@RequestBody Map<String, Object> request) {
        String workshopCode = (String) request.get("workshopCode");
        String workshopName = (String) request.get("workshopName");
        String description = (String) request.get("description");
        
        Long id = commandService.createWorkshop(workshopCode, workshopName, description);
        return ResponseEntity.ok(Map.of("workshopId", id));
    }

    @RequirePermission("dt:workshop:read")
    @GetMapping("/workshops")
    public ResponseEntity<?> getAllWorkshops(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        int limitedSize = Math.min(size, 100);
        return ResponseEntity.ok(queryService.getWorkshopsPaginated(page, limitedSize));
    }

    // ProductionLine endpoints
    @RequirePermission("dt:production-line:create")
    @PostMapping("/production-line")
    public ResponseEntity<Map<String, Object>> createProductionLine(@RequestBody Map<String, Object> request) {
        String lineCode = (String) request.get("lineCode");
        String lineName = (String) request.get("lineName");
        String workshopId = (String) request.get("workshopId");
        Integer capacity = (Integer) request.get("capacity");
        
        Long id = commandService.createProductionLine(lineCode, lineName, workshopId, capacity);
        return ResponseEntity.ok(Map.of("lineId", id));
    }

    @RequirePermission("dt:production-line:read")
    @GetMapping("/production-lines")
    public ResponseEntity<?> getAllProductionLines(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        int limitedSize = Math.min(size, 100);
        return ResponseEntity.ok(queryService.getProductionLinesPaginated(page, limitedSize));
    }

    // Alert endpoints
    @RequirePermission("dt:alert:create")
    @PostMapping("/alert")
    public ResponseEntity<Map<String, Object>> createAlert(@RequestBody Map<String, Object> request) {
        String deviceCode = (String) request.get("deviceCode");
        String workshopId = (String) request.get("workshopId");
        String level = (String) request.get("level");
        String type = (String) request.get("type");
        String title = (String) request.get("title");
        String description = (String) request.get("description");
        
        Alert.AlertLevel alertLevel = Alert.AlertLevel.valueOf(level.toUpperCase());
        Alert.AlertType alertType = Alert.AlertType.valueOf(type.toUpperCase());
        
        Long id = commandService.createAlert(deviceCode, workshopId, alertLevel, alertType, title, description);
        return ResponseEntity.ok(Map.of("alertId", id));
    }

    @RequirePermission("dt:alert:ack")
    @PostMapping("/alert/{id}/acknowledge")
    public ResponseEntity<Void> acknowledgeAlert(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        String acknowledgedBy = (String) request.get("acknowledgedBy");
        commandService.acknowledgeAlert(id, acknowledgedBy);
        return ResponseEntity.ok().build();
    }

    @RequirePermission("dt:alert:resolve")
    @PostMapping("/alert/{id}/resolve")
    public ResponseEntity<Void> resolveAlert(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        String solution = (String) request.get("solution");
        String resolvedBy = (String) request.get("resolvedBy");
        commandService.resolveAlert(id, solution, resolvedBy);
        return ResponseEntity.ok().build();
    }

    @RequirePermission("dt:alert:read")
    @GetMapping("/alerts/active")
    public ResponseEntity<?> getActiveAlerts() {
        return ResponseEntity.ok(queryService.getActiveAlerts());
    }

    // Statistics
    @RequirePermission("dt:stats:read")
    @GetMapping("/stats/summary")
    public ResponseEntity<?> getStatsSummary() {
        return ResponseEntity.ok(Map.of(
            "onlineDeviceCount", queryService.getOnlineDeviceCount(),
            "activeAlertCount", queryService.getActiveAlertCount(),
            "averageEfficiency", queryService.getAverageEfficiency()
        ));
    }
}