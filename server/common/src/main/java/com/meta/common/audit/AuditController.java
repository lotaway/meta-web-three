package com.meta.common.audit;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/audit")
public class AuditController {

    @Autowired
    private AuditLogService auditLogService;

    @GetMapping("/logs")
    public ResponseEntity<List<AuditLog>> getAllLogs() {
        List<AuditLog> logs = auditLogService.findAll();
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/logs/{id}")
    public ResponseEntity<AuditLog> getLogById(@PathVariable Long id) {
        AuditLog log = auditLogService.findById(id);
        if (log == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(log);
    }

    @GetMapping("/logs/username/{username}")
    public ResponseEntity<List<AuditLog>> getLogsByUsername(@PathVariable String username) {
        List<AuditLog> logs = auditLogService.findByUsername(username);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/logs/operation-type/{operationType}")
    public ResponseEntity<List<AuditLog>> getLogsByOperationType(@PathVariable String operationType) {
        List<AuditLog> logs = auditLogService.findByOperationType(operationType);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/logs/resource-type/{resourceType}")
    public ResponseEntity<List<AuditLog>> getLogsByResourceType(@PathVariable String resourceType) {
        List<AuditLog> logs = auditLogService.findByResourceType(resourceType);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/logs/time-range")
    public ResponseEntity<List<AuditLog>> getLogsByTimeRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        List<AuditLog> logs = auditLogService.findByTimeRange(startTime, endTime);
        return ResponseEntity.ok(logs);
    }

    @PostMapping("/logs/query")
    public ResponseEntity<List<AuditLog>> queryLogs(@RequestBody AuditLogQueryCondition condition) {
        List<AuditLog> logs = auditLogService.findByCondition(condition);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/logs/count")
    public ResponseEntity<Long> countLogs() {
        long count = auditLogService.count();
        return ResponseEntity.ok(count);
    }

    @DeleteMapping("/logs/cleanup")
    public ResponseEntity<String> cleanupLogs(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime beforeTime) {
        int deletedCount = auditLogService.cleanupBefore(beforeTime);
        return ResponseEntity.ok("Cleaned up " + deletedCount + " audit logs");
    }

    @DeleteMapping("/logs/clear")
    public ResponseEntity<String> clearAllLogs() {
        auditLogService.clearAll();
        return ResponseEntity.ok("All audit logs cleared");
    }

    @PostMapping("/logs")
    public ResponseEntity<AuditLog> createLog(@RequestBody AuditLog auditLog) {
        AuditLog saved = auditLogService.log(auditLog);
        return ResponseEntity.ok(saved);
    }
}
