package com.metawebthree.common.alert;

import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthComponent;
import org.springframework.boot.actuate.health.HealthEndpoint;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Real-time alerting service for exception tracing and monitoring
 */
@Service
public class AlertService {
    
    @Autowired
    private HealthEndpoint healthEndpoint;
    
    private final Map<String, AlertRecord> alertHistory = new HashMap<>();
    
    /**
     * Check system health and trigger alerts if necessary
     */
    public HealthCheckResult checkSystemHealth() {
        HealthComponent health = healthEndpoint.health();
        HealthCheckResult result = new HealthCheckResult();
        
        result.setTimestamp(LocalDateTime.now());
        result.setStatus(health.getStatus().getCode());
        
        if (!"UP".equals(health.getStatus().getCode())) {
            triggerAlert("SYSTEM_HEALTH_DOWN", "System health check failed: " + health.getStatus().getCode());
            result.setAlertTriggered(true);
        }
        
        return result;
    }
    
    /**
     * Record exception for tracing
     */
    public void recordException(String serviceName, String methodName, Exception ex) {
        AlertRecord record = new AlertRecord();
        record.setTimestamp(LocalDateTime.now());
        record.setServiceName(serviceName);
        record.setMethodName(methodName);
        record.setExceptionType(ex.getClass().getSimpleName());
        record.setExceptionMessage(ex.getMessage());
        record.setAlertType("EXCEPTION");
        
        String key = serviceName + ":" + methodName + ":" + System.currentTimeMillis();
        alertHistory.put(key, record);
        
        // Keep only last 1000 records
        if (alertHistory.size() > 1000) {
            alertHistory.entrySet().removeIf(entry -> 
                alertHistory.size() > 1000 && entry.getValue().getTimestamp()
                    .isBefore(LocalDateTime.now().minusHours(24))
            );
        }
    }
    
    /**
     * Trigger an alert with specified type and message
     */
    public void triggerAlert(String alertType, String message) {
        AlertRecord record = new AlertRecord();
        record.setTimestamp(LocalDateTime.now());
        record.setAlertType(alertType);
        record.setMessage(message);
        
        String key = alertType + ":" + System.currentTimeMillis();
        alertHistory.put(key, record);
        
        // Log alert for real-time monitoring
        System.err.println("[ALERT] " + alertType + ": " + message);
    }
    
    /**
     * Get recent alert history for tracing
     */
    public Map<String, AlertRecord> getRecentAlerts(int hours) {
        LocalDateTime since = LocalDateTime.now().minusHours(hours);
        Map<String, AlertRecord> recent = new HashMap<>();
        
        alertHistory.forEach((key, record) -> {
            if (record.getTimestamp().isAfter(since)) {
                recent.put(key, record);
            }
        });
        
        return recent;
    }
    
    /**
     * Clear old alert history
     */
    public void cleanupOldAlerts(int daysToKeep) {
        LocalDateTime cutoff = LocalDateTime.now().minusDays(daysToKeep);
        alertHistory.entrySet().removeIf(entry -> 
            entry.getValue().getTimestamp().isBefore(cutoff)
        );
    }
}
