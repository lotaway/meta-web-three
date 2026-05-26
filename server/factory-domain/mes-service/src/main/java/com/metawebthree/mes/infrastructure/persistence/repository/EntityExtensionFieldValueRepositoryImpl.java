package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class EntityExtensionFieldValueRepositoryImpl implements EntityExtensionFieldValueRepository {
    
    private final Map<Long, EntityExtensionFieldValue> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);
    
    @Override
    public Optional<EntityExtensionFieldValue> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }
    
    @Override
    public List<EntityExtensionFieldValue> findByEntityTypeAndEntityId(String entityType, Long entityId) {
        return storage.values().stream()
                .filter(v -> v.getEntityType().equals(entityType))
                .filter(v -> v.getEntityId().equals(entityId))
                .collect(Collectors.toList());
    }
    
    @Override
    public Optional<EntityExtensionFieldValue> findByEntityTypeAndEntityIdAndFieldCode(
            String entityType, Long entityId, String fieldCode) {
        return storage.values().stream()
                .filter(v -> v.getEntityType().equals(entityType))
                .filter(v -> v.getEntityId().equals(entityId))
                .filter(v -> v.getFieldCode().equals(fieldCode))
                .findFirst();
    }
    
    @Override
    public List<EntityExtensionFieldValue> saveAll(List<EntityExtensionFieldValue> values) {
        return values.stream()
                .map(this::save)
                .collect(Collectors.toList());
    }
    
    @Override
    public EntityExtensionFieldValue save(EntityExtensionFieldValue value) {
        // 查找是否已存在
        Optional<EntityExtensionFieldValue> existing = findByEntityTypeAndEntityIdAndFieldCode(
                value.getEntityType(), value.getEntityId(), value.getFieldCode());
        
        if (existing.isPresent()) {
            // 更新现有值
            existing.get().updateValue(value.getFieldValue());
            return existing.get();
        } else {
            // 新增
            if (value.getId() == null) {
                value.setId(idGen.getAndIncrement());
            }
            storage.put(value.getId(), value);
            return value;
        }
    }
    
    @Override
    public void deleteByEntityTypeAndEntityId(String entityType, Long entityId) {
        List<Long> toRemove = storage.values().stream()
                .filter(v -> v.getEntityType().equals(entityType))
                .filter(v -> v.getEntityId().equals(entityId))
                .map(EntityExtensionFieldValue::getId)
                .collect(Collectors.toList());
        toRemove.forEach(storage::remove);
    }
    
    @Override
    public void deleteByEntityTypeAndEntityIdAndFieldCode(String entityType, Long entityId, String fieldCode) {
        List<Long> toRemove = storage.values().stream()
                .filter(v -> v.getEntityType().equals(entityType))
                .filter(v -> v.getEntityId().equals(entityId))
                .filter(v -> v.getFieldCode().equals(fieldCode))
                .map(EntityExtensionFieldValue::getId)
                .collect(Collectors.toList());
        toRemove.forEach(storage::remove);
    }
}