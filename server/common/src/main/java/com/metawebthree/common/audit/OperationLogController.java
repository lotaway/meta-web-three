package com.metawebthree.common.audit;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.RequestParam;

@RestController
@RequestMapping("/api/audit/logs")
public class OperationLogController {

    @Autowired
    private OperationLogService operationLogService;

    @GetMapping
    public ResponseEntity<IPage<OperationLog>> getAllLogs(
            @RequestParam(defaultValue = "1") long current,
            @RequestParam(defaultValue = "20") long size) {
        Page<OperationLog> page = new Page<>(current, size);
        IPage<OperationLog> logs = operationLogService.page(page);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/{id}")
    public ResponseEntity<OperationLog> getLogById(@PathVariable Long id) {
        OperationLog log = operationLogService.getById(id);
        return ResponseEntity.ok(log);
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<List<OperationLog>> getLogsByUserId(@PathVariable Long userId) {
        List<OperationLog> logs = operationLogService.findByUserIdOrderByOperationTimeDesc(userId);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/operation/{operation}")
    public ResponseEntity<List<OperationLog>> getLogsByOperation(@PathVariable String operation) {
        List<OperationLog> logs = operationLogService.findByOperationOrderByOperationTimeDesc(operation);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/status/{status}")
    public ResponseEntity<List<OperationLog>> getLogsByStatus(@PathVariable String status) {
        List<OperationLog> logs = operationLogService.findByStatusOrderByOperationTimeDesc(status);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/entity/{entityType}/{entityId}")
    public ResponseEntity<List<OperationLog>> getLogsByEntity(
            @PathVariable String entityType,
            @PathVariable Long entityId) {
        List<OperationLog> logs = operationLogService.findByEntityTypeAndEntityIdOrderByOperationTimeDesc(entityType, entityId);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/timerange")
    public ResponseEntity<List<OperationLog>> getLogsByTimeRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        List<OperationLog> logs = operationLogService.findByOperationTimeBetweenOrderByOperationTimeDesc(startTime, endTime);
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/failures")
    public ResponseEntity<List<OperationLog>> getFailedOperations() {
        List<OperationLog> logs = operationLogService.findFailedOperations();
        return ResponseEntity.ok(logs);
    }

    @GetMapping("/statistics")
    public ResponseEntity<Map<String, Object>> getStatistics() {
        Map<String, Object> statistics = new HashMap<>();
        statistics.put("totalLogs", operationLogService.count());
        statistics.put("successCount", operationLogService.countByStatus("SUCCESS"));
        statistics.put("failureCount", operationLogService.countByStatus("FAILURE"));
        return ResponseEntity.ok(statistics);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteLog(@PathVariable Long id) {
        operationLogService.removeById(id);
        return ResponseEntity.noContent().build();
    }

    @DeleteMapping("/cleanup")
    public ResponseEntity<Map<String, String>> cleanupOldLogs(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDateTime beforeDate) {
        operationLogService.deleteOlderThan(beforeDate);
        Map<String, String> response = new HashMap<>();
        response.put("message", "Old logs cleaned up successfully");
        return ResponseEntity.ok(response);
    }
}
