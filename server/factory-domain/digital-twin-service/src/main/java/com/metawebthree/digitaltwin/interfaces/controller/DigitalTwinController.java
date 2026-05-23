package com.metawebthree.digitaltwin.interfaces.controller;

import com.metawebthree.digitaltwin.application.command.DigitalTwinCommandService;
import com.metawebthree.digitaltwin.application.query.DigitalTwinQueryService;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/digital-twin")
@CrossOrigin(origins = {"http://localhost:5173", "http://127.0.0.1:5173"})
public class DigitalTwinController {

    private final DigitalTwinCommandService commandService;
    private final DigitalTwinQueryService queryService;

    public DigitalTwinController(
            DigitalTwinCommandService commandService,
            DigitalTwinQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // Device endpoints
    @PostMapping("/device")
    public ResponseEntity<Map<String, Object>> registerDevice(@RequestBody Map<String, Object> request) {
        String deviceCode = (String) request.get("deviceCode");
        String deviceName = (String) request.get("deviceName");
        String deviceType = (String) request.get("deviceType");
        String workshopId = (String) request.get("workshopId");
        String productionLineId = (String) request.get("productionLineId");
        
        Long id = commandService.registerDevice(deviceCode, deviceName, deviceType, workshopId, productionLineId);
        return ResponseEntity.ok(Map.of("deviceId", id));
    }

    @PostMapping("/device/{deviceCode}/status")
    public ResponseEntity<Void> updateDeviceStatus(
            @PathVariable String deviceCode,
            @RequestBody Map<String, Object> request) {
        String status = (String) request.get("status");
        Device.DeviceStatus deviceStatus;
        try {
            deviceStatus = Device.DeviceStatus.valueOf(status.toUpperCase());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }
        commandService.updateDeviceStatus(deviceCode, deviceStatus);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/device/{deviceCode}/heartbeat")
    public ResponseEntity<Void> deviceHeartbeat(@PathVariable String deviceCode) {
        commandService.deviceHeartbeat(deviceCode);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/device/{deviceCode}/position")
    public ResponseEntity<Void> updateDevicePosition(
            @PathVariable String deviceCode,
            @RequestBody Map<String, Object> request) {
        Double x = ((Number) request.get("x")).doubleValue();
        Double y = ((Number) request.get("y")).doubleValue();
        Double z = ((Number) request.get("z")).doubleValue();
        Double rotation = request.get("rotation") != null ? 
            ((Number) request.get("rotation")).doubleValue() : 0.0;
        commandService.updateDevicePosition(deviceCode, x, y, z, rotation);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/device/{id}")
    public ResponseEntity<?> getDevice(@PathVariable Long id) {
        return queryService.getDeviceById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/devices")
    public ResponseEntity<?> getAllDevices(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        // Clamp page size to prevent abuse
        int limitedSize = Math.min(size, 100);
        return ResponseEntity.ok(queryService.getDevicesPaginated(page, limitedSize));
    }

    @GetMapping("/workshop/{workshopId}/devices")
    public ResponseEntity<?> getWorkshopDevices(@PathVariable String workshopId) {
        return ResponseEntity.ok(queryService.getWorkshopDevices(workshopId));
    }

    @GetMapping("/devices/online")
    public ResponseEntity<?> getOnlineDevices() {
        return ResponseEntity.ok(queryService.getOnlineDevices());
    }

    // Workshop endpoints
    @PostMapping("/workshop")
    public ResponseEntity<Map<String, Object>> createWorkshop(@RequestBody Map<String, Object> request) {
        String workshopCode = (String) request.get("workshopCode");
        String workshopName = (String) request.get("workshopName");
        String description = (String) request.get("description");
        
        Long id = commandService.createWorkshop(workshopCode, workshopName, description);
        return ResponseEntity.ok(Map.of("workshopId", id));
    }

    @GetMapping("/workshops")
    public ResponseEntity<?> getAllWorkshops(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        int limitedSize = Math.min(size, 100);
        return ResponseEntity.ok(queryService.getWorkshopsPaginated(page, limitedSize));
    }

    // ProductionLine endpoints
    @PostMapping("/production-line")
    public ResponseEntity<Map<String, Object>> createProductionLine(@RequestBody Map<String, Object> request) {
        String lineCode = (String) request.get("lineCode");
        String lineName = (String) request.get("lineName");
        String workshopId = (String) request.get("workshopId");
        Integer capacity = (Integer) request.get("capacity");
        
        Long id = commandService.createProductionLine(lineCode, lineName, workshopId, capacity);
        return ResponseEntity.ok(Map.of("lineId", id));
    }

    @GetMapping("/production-lines")
    public ResponseEntity<?> getAllProductionLines(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        int limitedSize = Math.min(size, 100);
        return ResponseEntity.ok(queryService.getProductionLinesPaginated(page, limitedSize));
    }

    // Alert endpoints
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

    @PostMapping("/alert/{id}/acknowledge")
    public ResponseEntity<Void> acknowledgeAlert(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        String acknowledgedBy = (String) request.get("acknowledgedBy");
        commandService.acknowledgeAlert(id, acknowledgedBy);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/alert/{id}/resolve")
    public ResponseEntity<Void> resolveAlert(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        String solution = (String) request.get("solution");
        String resolvedBy = (String) request.get("resolvedBy");
        commandService.resolveAlert(id, solution, resolvedBy);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/alerts/active")
    public ResponseEntity<?> getActiveAlerts() {
        return ResponseEntity.ok(queryService.getActiveAlerts());
    }

    // Statistics
    @GetMapping("/stats/summary")
    public ResponseEntity<?> getStatsSummary() {
        return ResponseEntity.ok(Map.of(
            "onlineDeviceCount", queryService.getOnlineDeviceCount(),
            "activeAlertCount", queryService.getActiveAlertCount(),
            "averageEfficiency", queryService.getAverageEfficiency()
        ));
    }
}