package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import java.util.List;
import java.util.Optional;

public interface EntityExtensionFieldValueRepository {
    
    Optional<EntityExtensionFieldValue> findById(Long id);
    
    List<EntityExtensionFieldValue> findByEntityTypeAndEntityId(String entityType, Long entityId);
    
    Optional<EntityExtensionFieldValue> findByEntityTypeAndEntityIdAndFieldCode(
            String entityType, Long entityId, String fieldCode);
    
    List<EntityExtensionFieldValue> saveAll(List<EntityExtensionFieldValue> values);
    
    EntityExtensionFieldValue save(EntityExtensionFieldValue value);
    
    void deleteByEntityTypeAndEntityId(String entityType, Long entityId);
    
    void deleteByEntityTypeAndEntityIdAndFieldCode(String entityType, Long entityId, String fieldCode);
}