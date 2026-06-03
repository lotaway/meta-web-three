package com.metawebthree.common.alert;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * REST controller for alert querying and management
 */
@RestController
@RequestMapping("/alert")
public class AlertController {
    
    @Autowired
    private AlertService alertService;
    
    /**
     * Get system health status
     */
    @GetMapping("/health")
    public ResponseEntity<?> getSystemHealth() {
        HealthCheckResult result = alertService.checkSystemHealth();
        return ResponseEntity.ok(result);
    }
    
    /**
     * Get recent alerts for tracing
     */
    @GetMapping("/recent")
    public ResponseEntity<?> getRecentAlerts(
            @RequestParam(value = "hours", defaultValue = "24") int hours) {
        Map<String, AlertRecord> alerts = alertService.getRecentAlerts(hours);
        return ResponseEntity.ok(alerts);
    }
    
    /**
     * Trigger a custom alert manually (for testing)
     */
    @PostMapping("/trigger")
    public ResponseEntity<?> triggerAlert(
            @RequestParam("type") String alertType,
            @RequestParam("message") String message) {
        alertService.triggerAlert(alertType, message);
        return ResponseEntity.ok().build();
    }
    
    /**
     * Cleanup old alert history
     */
    @DeleteMapping("/cleanup")
    public ResponseEntity<?> cleanupOldAlerts(
            @RequestParam(value = "days", defaultValue = "7") int daysToKeep) {
        alertService.cleanupOldAlerts(daysToKeep);
        return ResponseEntity.ok().build();
    }
}
