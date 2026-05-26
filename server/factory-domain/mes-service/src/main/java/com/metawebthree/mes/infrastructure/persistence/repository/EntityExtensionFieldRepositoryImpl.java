package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.EntityExtensionField;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class EntityExtensionFieldRepositoryImpl implements EntityExtensionFieldRepository {
    
    private final Map<Long, EntityExtensionField> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);
    
    @Override
    public EntityExtensionField findById(Long id) {
        return storage.get(id);
    }
    
    @Override
    public List<EntityExtensionField> findByEntityType(String entityType) {
        return storage.values().stream()
                .filter(f -> f.getEntityType().equals(entityType))
                .filter(f -> f.getStatus() == EntityExtensionField.FieldStatus.ACTIVE)
                .sorted(Comparator.comparing(EntityExtensionField::getSortOrder, Comparator.nullsLast(Comparator.naturalOrder())))
                .collect(Collectors.toList());
    }
    
    @Override
    public EntityExtensionField findByEntityTypeAndFieldCode(String entityType, String fieldCode) {
        return storage.values().stream()
                .filter(f -> f.getEntityType().equals(entityType))
                .filter(f -> f.getFieldCode().equals(fieldCode))
                .findFirst()
                .orElse(null);
    }
    
    @Override
    public EntityExtensionField save(EntityExtensionField field) {
        if (field.getId() == null) {
            field.setId(idGen.getAndIncrement());
        }
        storage.put(field.getId(), field);
        return field;
    }
    
    @Override
    public void delete(Long id) {
        storage.remove(id);
    }
    
    @Override
    public boolean existsByEntityTypeAndFieldCode(String entityType, String fieldCode) {
        return storage.values().stream()
                .anyMatch(f -> f.getEntityType().equals(entityType) && f.getFieldCode().equals(fieldCode));
    }
}