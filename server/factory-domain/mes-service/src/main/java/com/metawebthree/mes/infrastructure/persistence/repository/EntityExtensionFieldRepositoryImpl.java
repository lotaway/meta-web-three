package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.EntityExtensionField;
import com.metawebthree.mes.domain.entity.EntityExtensionField.FieldStatus;
import com.metawebthree.mes.domain.entity.EntityExtensionField.FieldType;
import com.metawebthree.mes.domain.repository.EntityExtensionFieldRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EntityExtensionFieldDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EntityExtensionFieldMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class EntityExtensionFieldRepositoryImpl implements EntityExtensionFieldRepository {
    
    @Autowired
    private EntityExtensionFieldMapper entityExtensionFieldMapper;
    
    @Override
    public Optional<EntityExtensionField> findById(Long id) {
        EntityExtensionFieldDO doObj = entityExtensionFieldMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<EntityExtensionField> findByEntityType(String entityType) {
        LambdaQueryWrapper<EntityExtensionFieldDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldDO::getEntityType, entityType)
               .eq(EntityExtensionFieldDO::getStatus, FieldStatus.ACTIVE.name())
               .orderByAsc(EntityExtensionFieldDO::getSortOrder);
        List<EntityExtensionFieldDO> doList = entityExtensionFieldMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public Optional<EntityExtensionField> findByEntityTypeAndFieldCode(String entityType, String fieldCode) {
        LambdaQueryWrapper<EntityExtensionFieldDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldDO::getEntityType, entityType)
               .eq(EntityExtensionFieldDO::getFieldCode, fieldCode);
        EntityExtensionFieldDO doObj = entityExtensionFieldMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public EntityExtensionField save(EntityExtensionField field) {
        EntityExtensionFieldDO doObj = toDO(field);
        if (field.getId() == null) {
            entityExtensionFieldMapper.insert(doObj);
            field.setId(doObj.getId());
        } else {
            entityExtensionFieldMapper.updateById(doObj);
        }
        return field;
    }
    
    @Override
    public void delete(Long id) {
        entityExtensionFieldMapper.deleteById(id);
    }
    
    @Override
    public boolean existsByEntityTypeAndFieldCode(String entityType, String fieldCode) {
        LambdaQueryWrapper<EntityExtensionFieldDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(EntityExtensionFieldDO::getEntityType, entityType)
               .eq(EntityExtensionFieldDO::getFieldCode, fieldCode);
        return entityExtensionFieldMapper.selectCount(wrapper) > 0;
    }
    
    // ========== DO 与 Entity 转换方法 ==========
    
    private EntityExtensionField toEntity(EntityExtensionFieldDO doObj) {
        if (doObj == null) {
            return null;
        }
        EntityExtensionField entity = new EntityExtensionField();
        entity.setId(doObj.getId());
        entity.setEntityType(doObj.getEntityType());
        entity.setFieldCode(doObj.getFieldCode());
        entity.setFieldName(doObj.getFieldName());
        entity.setFieldType(doObj.getFieldType() != null ? FieldType.valueOf(doObj.getFieldType()) : null);
        entity.setDefaultValue(doObj.getDefaultValue());
        entity.setRequired(doObj.getRequired());
        entity.setIsUnique(doObj.getIsUnique());
        entity.setValidationRule(doObj.getValidationRule());
        entity.setListVisible(doObj.getListVisible());
        entity.setSearchable(doObj.getSearchable());
        entity.setSortOrder(doObj.getSortOrder());
        entity.setFieldGroup(doObj.getFieldGroup());
        entity.setReferenceType(doObj.getReferenceType());
        entity.setReferenceEntity(doObj.getReferenceEntity());
        entity.setStatus(doObj.getStatus() != null ? FieldStatus.valueOf(doObj.getStatus()) : null);
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private EntityExtensionFieldDO toDO(EntityExtensionField entity) {
        if (entity == null) {
            return null;
        }
        EntityExtensionFieldDO doObj = new EntityExtensionFieldDO();
        doObj.setId(entity.getId());
        doObj.setEntityType(entity.getEntityType());
        doObj.setFieldCode(entity.getFieldCode());
        doObj.setFieldName(entity.getFieldName());
        doObj.setFieldType(entity.getFieldType() != null ? entity.getFieldType().name() : null);
        doObj.setDefaultValue(entity.getDefaultValue());
        doObj.setRequired(entity.getRequired());
        doObj.setIsUnique(entity.getIsUnique());
        doObj.setValidationRule(entity.getValidationRule());
        doObj.setListVisible(entity.getListVisible());
        doObj.setSearchable(entity.getSearchable());
        doObj.setSortOrder(entity.getSortOrder());
        doObj.setFieldGroup(entity.getFieldGroup());
        doObj.setReferenceType(entity.getReferenceType());
        doObj.setReferenceEntity(entity.getReferenceEntity());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        return doObj;
    }
}