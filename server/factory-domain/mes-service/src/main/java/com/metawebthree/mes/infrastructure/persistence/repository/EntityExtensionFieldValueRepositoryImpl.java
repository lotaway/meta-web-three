package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.EntityExtensionFieldValue;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldValueRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EntityExtensionFieldValueDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EntityExtensionFieldValueMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class EntityExtensionFieldValueRepositoryImpl implements EntityExtensionFieldValueRepository {
    
    @Autowired
    private EntityExtensionFieldValueMapper entityExtensionFieldValueMapper;
    
    @Override
    public Optional<EntityExtensionFieldValue> findById(Long id) {
        EntityExtensionFieldValueDO doObj = entityExtensionFieldValueMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<EntityExtensionFieldValue> findByEntityTypeAndEntityId(String entityType, Long entityId) {
        LambdaQueryWrapper<EntityExtensionFieldValueDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldValueDO::getEntityType, entityType)
               .eq(EntityExtensionFieldValueDO::getEntityId, entityId);
        List<EntityExtensionFieldValueDO> doList = entityExtensionFieldValueMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public Optional<EntityExtensionFieldValue> findByEntityTypeAndEntityIdAndFieldCode(
            String entityType, Long entityId, String fieldCode) {
        LambdaQueryWrapper<EntityExtensionFieldValueDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldValueDO::getEntityType, entityType)
               .eq(EntityExtensionFieldValueDO::getEntityId, entityId)
               .eq(EntityExtensionFieldValueDO::getFieldCode, fieldCode);
        EntityExtensionFieldValueDO doObj = entityExtensionFieldValueMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
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
        LambdaQueryWrapper<EntityExtensionFieldValueDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldValueDO::getEntityType, value.getEntityType())
               .eq(EntityExtensionFieldValueDO::getEntityId, value.getEntityId())
               .eq(EntityExtensionFieldValueDO::getFieldCode, value.getFieldCode());
        EntityExtensionFieldValueDO existingDO = entityExtensionFieldValueMapper.selectOne(wrapper);
        
        if (existingDO != null) {
            // 更新现有值
            existingDO.setFieldValue(value.getFieldValue());
            entityExtensionFieldValueMapper.updateById(existingDO);
            value.setId(existingDO.getId());
            return value;
        } else {
            // 新增
            EntityExtensionFieldValueDO doObj = toDO(value);
            entityExtensionFieldValueMapper.insert(doObj);
            value.setId(doObj.getId());
            return value;
        }
    }
    
    @Override
    public void deleteByEntityTypeAndEntityId(String entityType, Long entityId) {
        LambdaQueryWrapper<EntityExtensionFieldValueDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldValueDO::getEntityType, entityType)
               .eq(EntityExtensionFieldValueDO::getEntityId, entityId);
        entityExtensionFieldValueMapper.delete(wrapper);
    }
    
    @Override
    public void deleteByEntityTypeAndEntityIdAndFieldCode(String entityType, Long entityId, String fieldCode) {
        LambdaQueryWrapper<EntityExtensionFieldValueDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldValueDO::getEntityType, entityType)
               .eq(EntityExtensionFieldValueDO::getEntityId, entityId)
               .eq(EntityExtensionFieldValueDO::getFieldCode, fieldCode);
        entityExtensionFieldValueMapper.delete(wrapper);
    }
    
    // ========== DO 与 Entity 转换方法 ==========
    
    private EntityExtensionFieldValue toEntity(EntityExtensionFieldValueDO doObj) {
        if (doObj == null) {
            return null;
        }
        EntityExtensionFieldValue entity = new EntityExtensionFieldValue();
        entity.setId(doObj.getId());
        entity.setEntityType(doObj.getEntityType());
        entity.setEntityId(doObj.getEntityId());
        entity.setFieldCode(doObj.getFieldCode());
        entity.setFieldValue(doObj.getFieldValue());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private EntityExtensionFieldValueDO toDO(EntityExtensionFieldValue entity) {
        if (entity == null) {
            return null;
        }
        EntityExtensionFieldValueDO doObj = new EntityExtensionFieldValueDO();
        doObj.setId(entity.getId());
        doObj.setEntityType(entity.getEntityType());
        doObj.setEntityId(entity.getEntityId());
        doObj.setFieldCode(entity.getFieldCode());
        doObj.setFieldValue(entity.getFieldValue());
        return doObj;
    }
}