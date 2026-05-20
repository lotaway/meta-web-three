package com.metawebthree.aiwarehouse.domain.entity;

import java.time.LocalDateTime;

public class AICapability {
    private String capabilityId;
    private String capabilityName;
    private AICapabilityType type;
    private String endpoint;
    private String fallbackType;
    private String fallbackConfig;
    private Integer timeoutMs;
    private Integer retryCount;
    private Boolean enabled;
    private Integer priority;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum AICapabilityType {
        FORECASTING,
        RECOMMENDATION,
        RISK_SCORING,
        IMAGE_RECOGNITION,
        NLP,
        CUSTOM
    }

    public enum FallbackType {
        NONE,
        ALGORITHM,
        HUMAN,
        DEFAULT_VALUE,
        CACHE
    }

    public void register(String capabilityId, String capabilityName, AICapabilityType type,
                         String endpoint, FallbackType fallbackType, String fallbackConfig) {
        this.capabilityId = capabilityId;
        this.capabilityName = capabilityName;
        this.type = type;
        this.endpoint = endpoint;
        this.fallbackType = fallbackType.name();
        this.fallbackConfig = fallbackConfig;
        this.timeoutMs = 5000;
        this.retryCount = 3;
        this.enabled = true;
        this.priority = 100;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = this.createdAt;
    }

    public boolean isAvailable() {
        return enabled && endpoint != null && !endpoint.isEmpty();
    }

    public void enable() {
        this.enabled = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void disable() {
        this.enabled = false;
        this.updatedAt = LocalDateTime.now();
    }

    public String getCapabilityId() { return capabilityId; }
    public void setCapabilityId(String capabilityId) { this.capabilityId = capabilityId; }
    public String getCapabilityName() { return capabilityName; }
    public void setCapabilityName(String capabilityName) { this.capabilityName = capabilityName; }
    public AICapabilityType getType() { return type; }
    public void setType(AICapabilityType type) { this.type = type; }
    public String getEndpoint() { return endpoint; }
    public void setEndpoint(String endpoint) { this.endpoint = endpoint; }
    public String getFallbackType() { return fallbackType; }
    public void setFallbackType(String fallbackType) { this.fallbackType = fallbackType; }
    public String getFallbackConfig() { return fallbackConfig; }
    public void setFallbackConfig(String fallbackConfig) { this.fallbackConfig = fallbackConfig; }
    public Integer getTimeoutMs() { return timeoutMs; }
    public void setTimeoutMs(Integer timeoutMs) { this.timeoutMs = timeoutMs; }
    public Integer getRetryCount() { return retryCount; }
    public void setRetryCount(Integer retryCount) { this.retryCount = retryCount; }
    public Boolean getEnabled() { return enabled; }
    public void setEnabled(Boolean enabled) { this.enabled = enabled; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}