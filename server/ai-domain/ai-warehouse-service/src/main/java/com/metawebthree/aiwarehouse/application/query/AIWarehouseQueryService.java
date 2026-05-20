package com.metawebthree.aiwarehouse.application.query;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.entity.AIRequestRecord;
import com.metawebthree.aiwarehouse.domain.service.AIWarehouseDomainService;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class AIWarehouseQueryService {

    private final AIWarehouseDomainService domainService;

    public AIWarehouseQueryService(AIWarehouseDomainService domainService) {
        this.domainService = domainService;
    }

    public AICapabilityInfo getCapability(String capabilityId) {
        return domainService.getCapability(capabilityId)
            .map(this::toCapabilityInfo)
            .orElse(null);
    }

    public List<AICapabilityInfo> getCapabilitiesByType(String type) {
        AICapability.AICapabilityType capabilityType = 
            AICapability.AICapabilityType.valueOf(type.toUpperCase());
        return domainService.getCapabilitiesByType(capabilityType)
            .stream()
            .map(this::toCapabilityInfo)
            .collect(Collectors.toList());
    }

    public List<AICapabilityInfo> getAllEnabledCapabilities() {
        return domainService.getAllEnabledCapabilities()
            .stream()
            .map(this::toCapabilityInfo)
            .collect(Collectors.toList());
    }

    public List<AICapabilityInfo> getAllCapabilities() {
        return domainService.getAllCapabilities()
            .stream()
            .map(this::toCapabilityInfo)
            .collect(Collectors.toList());
    }

    public List<AIRequestRecordInfo> getRequestHistory(String capabilityId) {
        return domainService.getRequestHistory(capabilityId)
            .stream()
            .map(this::toRequestRecordInfo)
            .collect(Collectors.toList());
    }

    public List<AIRequestRecordInfo> getRecentRequests(int limit) {
        return domainService.getRecentRequests(limit)
            .stream()
            .map(this::toRequestRecordInfo)
            .collect(Collectors.toList());
    }

    public Map<String, Object> getStatistics() {
        List<AIRequestRecord> recentRequests = domainService.getRecentRequests(100);
        
        long total = recentRequests.size();
        long success = recentRequests.stream()
            .filter(r -> r.getStatus() == AIRequestRecord.AIRequestStatus.SUCCESS)
            .count();
        long fallback = recentRequests.stream()
            .filter(r -> r.getStatus() == AIRequestRecord.AIRequestStatus.FALLBACK_USED)
            .count();
        long failed = recentRequests.stream()
            .filter(r -> r.getStatus() == AIRequestRecord.AIRequestStatus.FAILED)
            .count();
        
        double avgExecutionTime = recentRequests.stream()
            .filter(r -> r.getExecutionTimeMs() != null)
            .mapToLong(AIRequestRecord::getExecutionTimeMs)
            .average()
            .orElse(0.0);
        
        return Map.of(
            "totalRequests", total,
            "successCount", success,
            "fallbackCount", fallback,
            "failedCount", failed,
            "successRate", total > 0 ? (double) success / total : 0.0,
            "fallbackRate", total > 0 ? (double) fallback / total : 0.0,
            "averageExecutionTimeMs", avgExecutionTime
        );
    }

    private AICapabilityInfo toCapabilityInfo(AICapability capability) {
        return new AICapabilityInfo(
            capability.getCapabilityId(),
            capability.getCapabilityName(),
            capability.getType().name(),
            capability.getEndpoint(),
            capability.getFallbackType(),
            capability.getFallbackConfig(),
            capability.getTimeoutMs(),
            capability.getRetryCount(),
            capability.getEnabled(),
            capability.getPriority()
        );
    }

    private AIRequestRecordInfo toRequestRecordInfo(AIRequestRecord record) {
        return new AIRequestRecordInfo(
            record.getId(),
            record.getRequestId(),
            record.getCapabilityId(),
            record.getCapabilityName(),
            record.getScene(),
            record.getCallerServiceName(),
            record.getStatus().name(),
            record.getFallbackUsed(),
            record.getFallbackReason(),
            record.getExecutionTimeMs(),
            record.getErrorMessage(),
            record.getCreatedAt() != null ? record.getCreatedAt().toString() : null,
            record.getCompletedAt() != null ? record.getCompletedAt().toString() : null
        );
    }

    public record AICapabilityInfo(
        String capabilityId,
        String capabilityName,
        String type,
        String endpoint,
        String fallbackType,
        String fallbackConfig,
        Integer timeoutMs,
        Integer retryCount,
        Boolean enabled,
        Integer priority
    ) {}

    public record AIRequestRecordInfo(
        Long id,
        String requestId,
        String capabilityId,
        String capabilityName,
        String scene,
        String callerServiceName,
        String status,
        String fallbackUsed,
        String fallbackReason,
        Long executionTimeMs,
        String errorMessage,
        String createdAt,
        String completedAt
    ) {}
}