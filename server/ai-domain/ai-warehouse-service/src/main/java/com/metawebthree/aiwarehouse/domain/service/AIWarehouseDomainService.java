package com.metawebthree.aiwarehouse.domain.service;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.entity.AIRequestRecord;
import com.metawebthree.aiwarehouse.domain.repository.AICapabilityRepository;
import com.metawebthree.aiwarehouse.domain.repository.AIRequestRecordRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
public class AIWarehouseDomainService implements IAIWarehouseDomainService {

    private final AICapabilityRepository capabilityRepository;
    private final AIRequestRecordRepository requestRecordRepository;

    public AIWarehouseDomainService(
            AICapabilityRepository capabilityRepository,
            AIRequestRecordRepository requestRecordRepository) {
        this.capabilityRepository = capabilityRepository;
        this.requestRecordRepository = requestRecordRepository;
    }

    @Override
    public AICapability registerCapability(String capabilityId, String capabilityName,
            AICapability.AICapabilityType type, String endpoint,
            AICapability.FallbackType fallbackType, String fallbackConfig) {
        
        if (capabilityRepository.existsByCapabilityId(capabilityId)) {
            throw new IllegalArgumentException("Capability already exists: " + capabilityId);
        }
        
        AICapability capability = new AICapability();
        capability.register(capabilityId, capabilityName, type, endpoint,
                           fallbackType, fallbackConfig);
        return capabilityRepository.save(capability);
    }

    public void updateCapability(String capabilityId, String endpoint,
                                  AICapability.FallbackType fallbackType, String fallbackConfig) {
        AICapability capability = capabilityRepository.findByCapabilityId(capabilityId)
            .orElseThrow(() -> new IllegalArgumentException("Capability not found: " + capabilityId));
        
        capability.setEndpoint(endpoint);
        if (fallbackType != null) {
            capability.setFallbackType(fallbackType.name());
        }
        if (fallbackConfig != null) {
            capability.setFallbackConfig(fallbackConfig);
        }
        capabilityRepository.save(capability);
    }

    public void enableCapability(String capabilityId) {
        capabilityRepository.findByCapabilityId(capabilityId)
            .ifPresent(AICapability::enable);
    }

    public void disableCapability(String capabilityId) {
        capabilityRepository.findByCapabilityId(capabilityId)
            .ifPresent(AICapability::disable);
    }

    public Optional<AICapability> getCapability(String capabilityId) {
        return capabilityRepository.findByCapabilityId(capabilityId);
    }

    public List<AICapability> getCapabilitiesByType(AICapability.AICapabilityType type) {
        return capabilityRepository.findByType(type);
    }

    public List<AICapability> getAllEnabledCapabilities() {
        return capabilityRepository.findByEnabledTrue();
    }

    public List<AICapability> getAllCapabilities() {
        return capabilityRepository.findAll();
    }

    public AIRequestRecord createRequestRecord(String capabilityId, String capabilityName,
            String scene, Long callerServiceId, String callerServiceName,
            String requestPayload) {
        
        String requestId = UUID.randomUUID().toString();
        AIRequestRecord record = new AIRequestRecord();
        record.create(requestId, capabilityId, capabilityName, scene,
                     callerServiceId, callerServiceName, requestPayload);
        return requestRecordRepository.save(record);
    }

    public void markRequestSuccess(Long recordId, String responsePayload, Long executionTimeMs) {
        requestRecordRepository.findById(recordId).ifPresent(record -> {
            record.completeSuccess(responsePayload, executionTimeMs);
            requestRecordRepository.save(record);
        });
    }

    public void markRequestFallback(Long recordId, String fallbackType, String fallbackReason,
                                    String responsePayload, Long executionTimeMs) {
        requestRecordRepository.findById(recordId).ifPresent(record -> {
            record.useFallback(fallbackType, fallbackReason, responsePayload, executionTimeMs);
            requestRecordRepository.save(record);
        });
    }

    public void markRequestFailed(Long recordId, String errorMessage) {
        requestRecordRepository.findById(recordId).ifPresent(record -> {
            record.fail(errorMessage);
            requestRecordRepository.save(record);
        });
    }

    public List<AIRequestRecord> getRequestHistory(String capabilityId) {
        return requestRecordRepository.findByCapabilityId(capabilityId);
    }

    public void removeCapability(String capabilityId) {
        AICapability capability = capabilityRepository.findByCapabilityId(capabilityId)
            .orElseThrow(() -> new IllegalArgumentException("Capability not found: " + capabilityId));
        capabilityRepository.delete(capabilityId);
    }

    public void deleteRequestRecord(Long id) {
        requestRecordRepository.deleteById(id);
    }

    public List<AIRequestRecord> getRecentRequests(int limit) {
        return requestRecordRepository.findTop100ByOrderByCreatedAtDesc();
    }
}