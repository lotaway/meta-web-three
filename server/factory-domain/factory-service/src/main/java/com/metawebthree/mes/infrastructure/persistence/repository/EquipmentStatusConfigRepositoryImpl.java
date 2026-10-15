package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.EquipmentStatusConfig;
import com.metawebthree.mes.domain.repository.EquipmentStatusConfigRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentStatusConfigDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentStatusConfigMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class EquipmentStatusConfigRepositoryImpl implements EquipmentStatusConfigRepository {
    private final EquipmentStatusConfigMapper mapper;
    private final ObjectMapper json;

    public EquipmentStatusConfigRepositoryImpl(EquipmentStatusConfigMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public Optional<EquipmentStatusConfig> findById(Long id) {
        QueryWrapper<EquipmentStatusConfigDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<EquipmentStatusConfig> findByEquipmentTypeId(Long equipmentTypeId) {
        QueryWrapper<EquipmentStatusConfigDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<EquipmentStatusConfig> findByEquipmentTypeIdAndIsActive(Long equipmentTypeId, Boolean isActive) {
        QueryWrapper<EquipmentStatusConfigDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.eq("is_active", isActive);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public Optional<EquipmentStatusConfig> findByEquipmentTypeIdAndStatusCode(Long equipmentTypeId, String statusCode) {
        QueryWrapper<EquipmentStatusConfigDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.eq("status_code", statusCode);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<EquipmentStatusConfig> findByEquipmentTypeIdAndIsInitial(Long equipmentTypeId, Boolean isInitial) {
        QueryWrapper<EquipmentStatusConfigDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.eq("is_initial", isInitial);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<EquipmentStatusConfig> findByStatusCategory(String statusCategory) {
        QueryWrapper<EquipmentStatusConfigDO> query = new QueryWrapper<>();
        query.eq("status_category", statusCategory);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public EquipmentStatusConfig save(EquipmentStatusConfig statusConfig) {
        return persist(statusConfig);
    }

    @Override
    public void update(EquipmentStatusConfig statusConfig) {
        if (statusConfig.getId() == null) throw new IllegalArgumentException("ID is required for update");
        persist(statusConfig);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private EquipmentStatusConfig persist(EquipmentStatusConfig entity) {
        EquipmentStatusConfigDO dto = new EquipmentStatusConfigDO();
        BeanUtils.copyProperties(entity, dto);
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("EquipmentStatusConfig not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private EquipmentStatusConfig toEntity(EquipmentStatusConfigDO dto) {
        EquipmentStatusConfig entity = new EquipmentStatusConfig();
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }
}