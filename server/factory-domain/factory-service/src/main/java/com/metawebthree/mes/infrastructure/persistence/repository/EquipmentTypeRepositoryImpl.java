package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.EquipmentType;
import com.metawebthree.mes.domain.repository.EquipmentTypeRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentTypeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentTypeMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class EquipmentTypeRepositoryImpl implements EquipmentTypeRepository {
    private final EquipmentTypeMapper mapper;
    private final ObjectMapper json;

    public EquipmentTypeRepositoryImpl(EquipmentTypeMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public Optional<EquipmentType> findById(Long id) {
        QueryWrapper<EquipmentTypeDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<EquipmentType> findByTypeCode(String typeCode) {
        QueryWrapper<EquipmentTypeDO> query = new QueryWrapper<>();
        query.eq("type_code", typeCode);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<EquipmentType> findByCategory(String category) {
        QueryWrapper<EquipmentTypeDO> query = new QueryWrapper<>();
        query.eq("category", category);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<EquipmentType> findByIsActive(Boolean isActive) {
        QueryWrapper<EquipmentTypeDO> query = new QueryWrapper<>();
        query.eq("is_active", isActive);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<EquipmentType> findAll() {
        QueryWrapper<EquipmentTypeDO> query = new QueryWrapper<>();
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public EquipmentType save(EquipmentType equipmentType) {
        return persist(equipmentType);
    }

    @Override
    public void update(EquipmentType equipmentType) {
        if (equipmentType.getId() == null) throw new IllegalArgumentException("ID is required for update");
        persist(equipmentType);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private EquipmentType persist(EquipmentType entity) {
        EquipmentTypeDO dto = new EquipmentTypeDO();
        BeanUtils.copyProperties(entity, dto);
        try { dto.setAttributes(json.writeValueAsString(entity.getAttributes())); }
        catch (JsonProcessingException e) { throw new IllegalArgumentException("Invalid attributes JSON", e); }
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("EquipmentType not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private EquipmentType toEntity(EquipmentTypeDO dto) {
        EquipmentType entity = new EquipmentType();
        BeanUtils.copyProperties(dto, entity);
        try {
            entity.setAttributes(dto.getAttributes() == null ? null : json.readValue(dto.getAttributes(), new TypeReference<List<EquipmentType.EquipmentTypeAttribute>>() {}));
        } catch (JsonProcessingException e) { throw new IllegalStateException("Stored attributes JSON is invalid", e); }
        return entity;
    }
}