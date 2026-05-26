package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import java.util.List;
import java.util.Optional;

/**
 * 实体扩展字段值仓储接口
 */
public interface EntityExtensionFieldValueRepository {
    
    /**
     * 根据ID查询
     */
    Optional<EntityExtensionFieldValue> findById(Long id);
    
    /**
     * 根据实体类型和实体ID查询所有扩展字段值
     */
    List<EntityExtensionFieldValue> findByEntityTypeAndEntityId(String entityType, Long entityId);
    
    /**
     * 根据实体类型、实体ID和字段编码查询
     */
    Optional<EntityExtensionFieldValue> findByEntityTypeAndEntityIdAndFieldCode(
            String entityType, Long entityId, String fieldCode);
    
    /**
     * 批量保存扩展字段值
     */
    List<EntityExtensionFieldValue> saveAll(List<EntityExtensionFieldValue> values);
    
    /**
     * 保存扩展字段值
     */
    EntityExtensionFieldValue save(EntityExtensionFieldValue value);
    
    /**
     * 删除实体所有扩展字段值
     */
    void deleteByEntityTypeAndEntityId(String entityType, Long entityId);
    
    /**
     * 删除指定扩展字段值
     */
    void deleteByEntityTypeAndEntityIdAndFieldCode(String entityType, Long entityId, String fieldCode);
}