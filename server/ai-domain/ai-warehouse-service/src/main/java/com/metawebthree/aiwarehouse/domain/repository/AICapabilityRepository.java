package com.metawebthree.aiwarehouse.domain.repository;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import java.util.List;
import java.util.Optional;

public interface AICapabilityRepository {
    AICapability save(AICapability capability);
    Optional<AICapability> findById(String capabilityId);
    Optional<AICapability> findByCapabilityId(String capabilityId);
    List<AICapability> findByType(AICapability.AICapabilityType type);
    List<AICapability> findByEnabledTrue();
    List<AICapability> findAll();
    void delete(String capabilityId);
    boolean existsByCapabilityId(String capabilityId);
}