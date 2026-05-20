package com.metawebthree.aiwarehouse.domain.entity;

import java.time.LocalDateTime;

public class AIRequestRecord {
    private Long id;
    private String requestId;
    private String capabilityId;
    private String capabilityName;
    private String scene;
    private Long callerServiceId;
    private String callerServiceName;
    private String requestPayload;
    private String responsePayload;
    private AIRequestStatus status;
    private String fallbackUsed;
    private String fallbackReason;
    private Long executionTimeMs;
    private String errorMessage;
    private LocalDateTime createdAt;
    private LocalDateTime completedAt;

    public enum AIRequestStatus {
        PENDING,
        PROCESSING,
        SUCCESS,
        FALLBACK_USED,
        FAILED
    }

    public void create(String requestId, String capabilityId, String capabilityName,
                       String scene, Long callerServiceId, String callerServiceName,
                       String requestPayload) {
        this.requestId = requestId;
        this.capabilityId = capabilityId;
        this.capabilityName = capabilityName;
        this.scene = scene;
        this.callerServiceId = callerServiceId;
        this.callerServiceName = callerServiceName;
        this.requestPayload = requestPayload;
        this.status = AIRequestStatus.PENDING;
        this.createdAt = LocalDateTime.now();
    }

    public void startProcessing() {
        this.status = AIRequestStatus.PROCESSING;
    }

    public void completeSuccess(String responsePayload, Long executionTimeMs) {
        this.status = AIRequestStatus.SUCCESS;
        this.responsePayload = responsePayload;
        this.executionTimeMs = executionTimeMs;
        this.completedAt = LocalDateTime.now();
    }

    public void useFallback(String fallbackType, String fallbackReason,
                           String responsePayload, Long executionTimeMs) {
        this.status = AIRequestStatus.FALLBACK_USED;
        this.fallbackUsed = fallbackType;
        this.fallbackReason = fallbackReason;
        this.responsePayload = responsePayload;
        this.executionTimeMs = executionTimeMs;
        this.completedAt = LocalDateTime.now();
    }

    public void fail(String errorMessage) {
        this.status = AIRequestStatus.FAILED;
        this.errorMessage = errorMessage;
        this.completedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRequestId() { return requestId; }
    public void setRequestId(String requestId) { this.requestId = requestId; }
    public String getCapabilityId() { return capabilityId; }
    public void setCapabilityId(String capabilityId) { this.capabilityId = capabilityId; }
    public String getCapabilityName() { return capabilityName; }
    public void setCapabilityName(String capabilityName) { this.capabilityName = capabilityName; }
    public String getScene() { return scene; }
    public void setScene(String scene) { this.scene = scene; }
    public Long getCallerServiceId() { return callerServiceId; }
    public void setCallerServiceId(Long callerServiceId) { this.callerServiceId = callerServiceId; }
    public String getCallerServiceName() { return callerServiceName; }
    public void setCallerServiceName(String callerServiceName) { this.callerServiceName = callerServiceName; }
    public String getRequestPayload() { return requestPayload; }
    public void setRequestPayload(String requestPayload) { this.requestPayload = requestPayload; }
    public String getResponsePayload() { return responsePayload; }
    public void setResponsePayload(String responsePayload) { this.responsePayload = responsePayload; }
    public AIRequestStatus getStatus() { return status; }
    public void setStatus(AIRequestStatus status) { this.status = status; }
    public String getFallbackUsed() { return fallbackUsed; }
    public void setFallbackUsed(String fallbackUsed) { this.fallbackUsed = fallbackUsed; }
    public String getFallbackReason() { return fallbackReason; }
    public void setFallbackReason(String fallbackReason) { this.fallbackReason = fallbackReason; }
    public Long getExecutionTimeMs() { return executionTimeMs; }
    public void setExecutionTimeMs(Long executionTimeMs) { this.executionTimeMs = executionTimeMs; }
    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getCompletedAt() { return completedAt; }
    public void setCompletedAt(LocalDateTime completedAt) { this.completedAt = completedAt; }
}