package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.EquipmentStatusTransition;
import com.metawebthree.mes.domain.repository.EquipmentStatusTransitionRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.EquipmentStatusTransitionDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.EquipmentStatusTransitionMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class EquipmentStatusTransitionRepositoryImpl implements EquipmentStatusTransitionRepository {
    private final EquipmentStatusTransitionMapper mapper;
    private final ObjectMapper json;

    public EquipmentStatusTransitionRepositoryImpl(EquipmentStatusTransitionMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public Optional<EquipmentStatusTransition> findById(Long id) {
        QueryWrapper<EquipmentStatusTransitionDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<EquipmentStatusTransition> findByEquipmentTypeId(Long equipmentTypeId) {
        QueryWrapper<EquipmentStatusTransitionDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<EquipmentStatusTransition> findByEquipmentTypeIdAndIsActive(Long equipmentTypeId, Boolean isActive) {
        QueryWrapper<EquipmentStatusTransitionDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.eq("is_active", isActive);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<EquipmentStatusTransition> findByEquipmentTypeIdAndFromStatusCode(Long equipmentTypeId, String fromStatusCode) {
        QueryWrapper<EquipmentStatusTransitionDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.eq("from_status_code", fromStatusCode);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public Optional<EquipmentStatusTransition> findByEquipmentTypeIdAndFromStatusCodeAndToStatusCode(Long equipmentTypeId, String fromStatusCode, String toStatusCode) {
        QueryWrapper<EquipmentStatusTransitionDO> query = new QueryWrapper<>();
        query.eq("equipment_type_id", equipmentTypeId);
        query.eq("from_status_code", fromStatusCode);
        query.eq("to_status_code", toStatusCode);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public EquipmentStatusTransition save(EquipmentStatusTransition transition) {
        return persist(transition);
    }

    @Override
    public void update(EquipmentStatusTransition transition) {
        if (transition.getId() == null) throw new IllegalArgumentException("ID is required for update");
        persist(transition);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private EquipmentStatusTransition persist(EquipmentStatusTransition entity) {
        EquipmentStatusTransitionDO dto = new EquipmentStatusTransitionDO();
        BeanUtils.copyProperties(entity, dto);
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("EquipmentStatusTransition not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private EquipmentStatusTransition toEntity(EquipmentStatusTransitionDO dto) {
        EquipmentStatusTransition entity = new EquipmentStatusTransition();
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }
}