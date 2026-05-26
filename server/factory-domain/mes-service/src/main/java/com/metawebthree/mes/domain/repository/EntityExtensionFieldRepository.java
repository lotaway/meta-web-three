package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EntityExtensionField;
import java.util.List;

/**
 * 实体扩展字段仓储接口
 */
public interface EntityExtensionFieldRepository {
    
    /**
     * 根据ID查询
     */
    EntityExtensionField findById(Long id);
    
    /**
     * 根据实体类型查询所有启用的扩展字段
     */
    List<EntityExtensionField> findByEntityType(String entityType);
    
    /**
     * 根据实体类型和字段编码查询
     */
    EntityExtensionField findByEntityTypeAndFieldCode(String entityType, String fieldCode);
    
    /**
     * 保存扩展字段定义
     */
    EntityExtensionField save(EntityExtensionField field);
    
    /**
     * 删除扩展字段定义
     */
    void delete(Long id);
    
    /**
     * 检查字段编码是否已存在
     */
    boolean existsByEntityTypeAndFieldCode(String entityType, String fieldCode);
}