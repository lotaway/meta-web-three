package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EntityExtensionField;
import java.util.List;
import java.util.Optional;

public interface EntityExtensionFieldRepository {
    
    Optional<EntityExtensionField> findById(Long id);
    
    List<EntityExtensionField> findByEntityType(String entityType);
    
    Optional<EntityExtensionField> findByEntityTypeAndFieldCode(String entityType, String fieldCode);
    
    EntityExtensionField save(EntityExtensionField field);
    
    void delete(Long id);
    
    boolean existsByEntityTypeAndFieldCode(String entityType, String fieldCode);
}