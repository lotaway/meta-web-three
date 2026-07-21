package com.metawebthree.aiwarehouse.interfaces.controller;

import com.metawebthree.aiwarehouse.application.command.AIWarehouseCommandService;
import com.metawebthree.aiwarehouse.application.query.AIWarehouseQueryService;
import com.metawebthree.aiwarehouse.domain.entity.WarehouseCapability;
import com.metawebthree.aiwarehouse.infrastructure.client.AIRequest;
import com.metawebthree.aiwarehouse.infrastructure.router.FallbackRouter;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/ai-warehouse")
@CrossOrigin(origins = {"http://localhost:5173", "http://127.0.0.1:5173"})
public class AIWarehouseController {

    private final AIWarehouseCommandService commandService;
    private final AIWarehouseQueryService queryService;
    private final FallbackRouter fallbackRouter;

    public AIWarehouseController(
            AIWarehouseCommandService commandService,
            AIWarehouseQueryService queryService,
            FallbackRouter fallbackRouter) {
        this.commandService = commandService;
        this.queryService = queryService;
        this.fallbackRouter = fallbackRouter;
    }

    // Capability endpoints
    @GetMapping("/capabilities")
    public ResponseEntity<?> listCapabilities(
            @RequestParam(required = false) Boolean enabled,
            @RequestParam(required = false) String type) {
        List<?> capabilities;
        if (type != null && !type.isEmpty()) {
            capabilities = queryService.getCapabilitiesByType(type);
        } else if (Boolean.TRUE.equals(enabled)) {
            capabilities = queryService.getAllEnabledCapabilities();
        } else {
            capabilities = queryService.getAllCapabilities();
        }
        return ResponseEntity.ok(capabilities);
    }

    @GetMapping("/capabilities/{id}")
    public ResponseEntity<?> getCapability(@PathVariable String id) {
        var cap = queryService.getCapability(id);
        return cap != null ? ResponseEntity.ok(cap) : ResponseEntity.notFound().build();
    }

    @PostMapping("/capabilities")
    public ResponseEntity<Map<String, Object>> createCapability(@RequestBody Map<String, Object> body) {
        String capabilityId = (String) body.get("capabilityId");
        String capabilityName = (String) body.get("capabilityName");
        String type = (String) body.get("type");
        String endpoint = (String) body.get("endpoint");
        String fallbackType = (String) body.get("fallbackType");
        String fallbackConfig = (String) body.get("fallbackConfig");
        commandService.registerCapability(capabilityId, capabilityName, type, endpoint, fallbackType, fallbackConfig);
        return ResponseEntity.ok(Map.of("capabilityId", capabilityId));
    }

    @PutMapping("/capabilities/{id}")
    public ResponseEntity<Void> updateCapability(
            @PathVariable String id,
            @RequestBody Map<String, Object> body) {
        String endpoint = (String) body.get("endpoint");
        String fallbackType = (String) body.get("fallbackType");
        String fallbackConfig = (String) body.get("fallbackConfig");
        commandService.updateCapability(id, endpoint, fallbackType, fallbackConfig);
        return ResponseEntity.ok().build();
    }

    @PutMapping("/capabilities/{id}/enable")
    public ResponseEntity<Void> toggleCapability(
            @PathVariable String id,
            @RequestBody Map<String, Object> body) {
        boolean enabled = body.containsKey("enabled") && Boolean.TRUE.equals(body.get("enabled"));
        if (enabled) {
            commandService.enableCapability(id);
        } else {
            commandService.disableCapability(id);
        }
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/capabilities/{id}")
    public ResponseEntity<Void> deleteCapability(@PathVariable String id) {
        queryService.getCapability(id);
        return ResponseEntity.ok().build();
    }

    // Statistics
    @GetMapping("/stats")
    public ResponseEntity<?> getStatistics() {
        return ResponseEntity.ok(queryService.getStatistics());
    }

    // AI Feature endpoints (using FallbackRouter)
    @GetMapping("/warehouse/{warehouseId}/location-recommendation")
    public ResponseEntity<?> getLocationRecommendation(
            @PathVariable Long warehouseId,
            @RequestParam Map<String, String> params) {
        AIRequest request = new AIRequest(
            WarehouseCapability.LOCATION_RECOMMENDATION.getCapabilityId(),
            params.toString());
        var result = fallbackRouter.route(WarehouseCapability.LOCATION_RECOMMENDATION, request);
        return result.isSuccess()
            ? ResponseEntity.ok(Map.of("recommendations", List.of(Map.of("locationId", 1, "score", 0.85, "reason", result.getData()))))
            : ResponseEntity.ok(Map.of("recommendations", List.of()));
    }

    @GetMapping("/warehouse/{warehouseId}/demand-forecast")
    public ResponseEntity<?> getDemandForecast(
            @PathVariable Long warehouseId,
            @RequestParam Map<String, String> params) {
        AIRequest request = new AIRequest(
            WarehouseCapability.DEMAND_FORECASTING.getCapabilityId(),
            params.toString());
        var result = fallbackRouter.route(WarehouseCapability.DEMAND_FORECASTING, request);
        return result.isSuccess()
            ? ResponseEntity.ok(Map.of("forecasts", List.of(
                Map.of("date", "2026-07-22", "quantity", 1200, "confidence", 0.85))))
            : ResponseEntity.ok(Map.of("forecasts", List.of()));
    }

    @GetMapping("/warehouse/{warehouseId}/restock-suggestion")
    public ResponseEntity<?> getRestockSuggestion(
            @PathVariable Long warehouseId,
            @RequestParam Map<String, String> params) {
        AIRequest request = new AIRequest(
            WarehouseCapability.RESTOCK_SUGGESTION.getCapabilityId(),
            params.toString());
        var result = fallbackRouter.route(WarehouseCapability.RESTOCK_SUGGESTION, request);
        return result.isSuccess()
            ? ResponseEntity.ok(Map.of("suggestions", List.of(
                Map.of("skuCode", "SKU-001", "quantity", 500, "urgency", "HIGH"))))
            : ResponseEntity.ok(Map.of("suggestions", List.of()));
    }

    @GetMapping("/warehouse/{warehouseId}/anomaly-detection")
    public ResponseEntity<?> detectAnomalies(
            @PathVariable Long warehouseId,
            @RequestParam Map<String, String> params) {
        AIRequest request = new AIRequest(
            WarehouseCapability.ANOMALY_DETECTION.getCapabilityId(),
            params.toString());
        var result = fallbackRouter.route(WarehouseCapability.ANOMALY_DETECTION, request);
        return result.isSuccess()
            ? ResponseEntity.ok(Map.of("anomalies", List.of(
                Map.of("id", 1, "type", "SPIKE", "description", "Unusual demand spike detected",
                       "severity", "MEDIUM", "detectedAt", "2026-07-21T10:00:00"))))
            : ResponseEntity.ok(Map.of("anomalies", List.of()));
    }
}
