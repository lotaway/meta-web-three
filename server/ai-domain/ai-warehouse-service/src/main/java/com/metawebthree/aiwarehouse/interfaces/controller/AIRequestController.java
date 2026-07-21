package com.metawebthree.aiwarehouse.interfaces.controller;

import com.metawebthree.aiwarehouse.application.command.AIWarehouseCommandService;
import com.metawebthree.aiwarehouse.application.query.AIWarehouseQueryService;
import com.metawebthree.aiwarehouse.domain.entity.AIRequestRecord;
import com.metawebthree.aiwarehouse.domain.service.AIWarehouseDomainService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/ai-warehouse/requests")
public class AIRequestController {

    private static final Logger log = LoggerFactory.getLogger(AIRequestController.class);

    private final AIWarehouseQueryService queryService;
    private final AIWarehouseDomainService domainService;

    public AIRequestController(
            AIWarehouseQueryService queryService,
            AIWarehouseDomainService domainService) {
        this.queryService = queryService;
        this.domainService = domainService;
    }

    @GetMapping
    public ResponseEntity<Map<String, Object>> listRequests(
            @RequestParam(required = false, defaultValue = "1") int pageNum,
            @RequestParam(required = false, defaultValue = "20") int pageSize,
            @RequestParam(required = false) String capabilityType,
            @RequestParam(required = false) String status) {
        List<AIRequestRecord> all = domainService.getRecentRequests(100);
        List<Map<String, Object>> list = all.stream()
                .filter(r -> capabilityType == null || capabilityType.isEmpty()
                        || capabilityType.equals(r.getCapabilityId()))
                .filter(r -> status == null || status.isEmpty()
                        || status.equalsIgnoreCase(r.getStatus().name()))
                .map(this::toRequestMap)
                .collect(Collectors.toList());
        return ResponseEntity.ok(Map.of("list", list, "total", (long) list.size()));
    }

    @GetMapping("/{id}")
    public ResponseEntity<?> getRequest(@PathVariable Long id) {
        return domainService.getRequestHistory("").stream()
                .filter(r -> r.getId().equals(id))
                .findFirst()
                .map(r -> ResponseEntity.ok(toRequestMap(r)))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<Map<String, Object>> createRequest(@RequestBody Map<String, Object> body) {
        String capabilityType = (String) body.get("capabilityType");
        String requestData = (String) body.get("requestData");
        Long warehouseId = body.get("warehouseId") != null
                ? ((Number) body.get("warehouseId")).longValue() : 0L;
        String warehouseName = (String) body.get("warehouseName");

        AIRequestRecord record = domainService.createRequestRecord(
                capabilityType, capabilityType, "admin",
                warehouseId, warehouseName, requestData);
        return ResponseEntity.ok(Map.of("id", record.getId()));
    }

    @PutMapping("/{id}")
    public ResponseEntity<Void> updateRequest(
            @PathVariable Long id,
            @RequestBody Map<String, Object> body) {
        String responseData = (String) body.get("responseData");
        String statusStr = (String) body.get("status");
        if (responseData != null) {
            domainService.markRequestSuccess(id, responseData, 0L);
        }
        if (statusStr != null) {
            try {
                AIRequestRecord.AIRequestStatus s = AIRequestRecord.AIRequestStatus.valueOf(statusStr.toUpperCase());
                if (s == AIRequestRecord.AIRequestStatus.FAILED) {
                    domainService.markRequestFailed(id, "Manual fail");
                }
            } catch (Exception e) {
                log.warn("Failed to update request status {} for id {}: {}", statusStr, id, e.getMessage());
            }
        }
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteRequest(@PathVariable Long id) {
        domainService.deleteRequestRecord(id);
        return ResponseEntity.ok().build();
    }

    private Map<String, Object> toRequestMap(AIRequestRecord r) {
        return Map.of(
            "id", r.getId(),
            "warehouseId", r.getCallerServiceId() != null ? r.getCallerServiceId() : 0,
            "warehouseName", r.getCallerServiceName() != null ? r.getCallerServiceName() : "",
            "capabilityType", r.getCapabilityId() != null ? r.getCapabilityId() : "",
            "requestData", r.getRequestPayload() != null ? r.getRequestPayload() : "",
            "responseData", r.getResponsePayload() != null ? r.getResponsePayload() : "",
            "status", r.getStatus() != null ? r.getStatus().name() : "PENDING",
            "createdAt", r.getCreatedAt() != null ? r.getCreatedAt().toString() : "",
            "updatedAt", r.getCompletedAt() != null ? r.getCompletedAt().toString() : ""
        );
    }
}
