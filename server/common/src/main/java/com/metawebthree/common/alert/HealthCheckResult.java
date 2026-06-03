package com.metawebthree.common.alert;

import java.time.LocalDateTime;

/**
 * Health check result containing alert status
 */
public class HealthCheckResult {
    private LocalDateTime timestamp;
    private String status;
    private boolean alertTriggered;
    private String details;
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public String getStatus() {
        return status;
    }
    
    public void setStatus(String status) {
        this.status = status;
    }
    
    public boolean isAlertTriggered() {
        return alertTriggered;
    }
    
    public void setAlertTriggered(boolean alertTriggered) {
        this.alertTriggered = alertTriggered;
    }
    
    public String getDetails() {
        return details;
    }
    
    public void setDetails(String details) {
        this.details = details;
    }
}
