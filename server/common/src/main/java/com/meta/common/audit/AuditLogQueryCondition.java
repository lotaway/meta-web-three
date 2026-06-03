package com.meta.common.audit;

import java.time.LocalDateTime;

public class AuditLogQueryCondition {

    private Long userId;
    private String username;
    private String operationType;
    private String resourceType;
    private String resourceId;
    private String result;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private String ipAddress;
    private String requestUrl;

    public AuditLogQueryCondition() {
    }

    public boolean matches(AuditLog log) {
        if (userId != null && !userId.equals(log.getUserId())) {
            return false;
        }
        if (username != null && !username.equals(log.getUsername())) {
            return false;
        }
        if (operationType != null && !operationType.equals(log.getOperationType())) {
            return false;
        }
        if (resourceType != null && !resourceType.equals(log.getResourceType())) {
            return false;
        }
        if (resourceId != null && !resourceId.equals(log.getResourceId())) {
            return false;
        }
        if (result != null && !result.equals(log.getResult())) {
            return false;
        }
        if (startTime != null && log.getOperationTime().isBefore(startTime)) {
            return false;
        }
        if (endTime != null && log.getOperationTime().isAfter(endTime)) {
            return false;
        }
        if (ipAddress != null && !ipAddress.equals(log.getIpAddress())) {
            return false;
        }
        if (requestUrl != null && !requestUrl.equals(log.getRequestUrl())) {
            return false;
        }
        return true;
    }

    public Long getUserId() {
        return userId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getOperationType() {
        return operationType;
    }

    public void setOperationType(String operationType) {
        this.operationType = operationType;
    }

    public String getResourceType() {
        return resourceType;
    }

    public void setResourceType(String resourceType) {
        this.resourceType = resourceType;
    }

    public String getResourceId() {
        return resourceId;
    }

    public void setResourceId(String resourceId) {
        this.resourceId = resourceId;
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public LocalDateTime getStartTime() {
        return startTime;
    }

    public void setStartTime(LocalDateTime startTime) {
        this.startTime = startTime;
    }

    public LocalDateTime getEndTime() {
        return endTime;
    }

    public void setEndTime(LocalDateTime endTime) {
        this.endTime = endTime;
    }

    public String getIpAddress() {
        return ipAddress;
    }

    public void setIpAddress(String ipAddress) {
        this.ipAddress = ipAddress;
    }

    public String getRequestUrl() {
        return requestUrl;
    }

    public void setRequestUrl(String requestUrl) {
        this.requestUrl = requestUrl;
    }
}
