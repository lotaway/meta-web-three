package com.metawebthree.aiwarehouse.infrastructure.persistence.repository;

import com.metawebthree.aiwarehouse.domain.entity.AICapability;
import com.metawebthree.aiwarehouse.domain.repository.AICapabilityRepository;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class AICapabilityRepositoryImpl implements AICapabilityRepository {

    private final ConcurrentHashMap<String, AICapability> store = new ConcurrentHashMap<>();

    @Override
    public AICapability save(AICapability capability) {
        store.put(capability.getCapabilityId(), capability);
        return capability;
    }

    @Override
    public Optional<AICapability> findById(String capabilityId) {
        return Optional.ofNullable(store.get(capabilityId));
    }

    @Override
    public Optional<AICapability> findByCapabilityId(String capabilityId) {
        return Optional.ofNullable(store.get(capabilityId));
    }

    @Override
    public List<AICapability> findByType(AICapability.AICapabilityType type) {
        return store.values().stream()
                .filter(c -> c.getType() == type)
                .collect(Collectors.toList());
    }

    @Override
    public List<AICapability> findByEnabledTrue() {
        return store.values().stream()
                .filter(AICapability::getEnabled)
                .collect(Collectors.toList());
    }

    @Override
    public List<AICapability> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void delete(String capabilityId) {
        store.remove(capabilityId);
    }

    @Override
    public boolean existsByCapabilityId(String capabilityId) {
        return store.containsKey(capabilityId);
    }
}
