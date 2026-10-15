package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.AndonType;
import com.metawebthree.mes.domain.repository.AndonTypeRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.AndonTypeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.AndonTypeMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class AndonTypeRepositoryImpl implements AndonTypeRepository {
    private final AndonTypeMapper mapper;
    private final ObjectMapper json;

    public AndonTypeRepositoryImpl(AndonTypeMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public Optional<AndonType> findById(Long id) {
        QueryWrapper<AndonTypeDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<AndonType> findByTypeCode(String typeCode) {
        QueryWrapper<AndonTypeDO> query = new QueryWrapper<>();
        query.eq("type_code", typeCode);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<AndonType> findByCategory(AndonType.AndonCategory category) {
        QueryWrapper<AndonTypeDO> query = new QueryWrapper<>();
        query.eq("category", category);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonType> findByStatus(AndonType.AndonStatus status) {
        QueryWrapper<AndonTypeDO> query = new QueryWrapper<>();
        query.eq("status", status);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonType> findAll() {
        QueryWrapper<AndonTypeDO> query = new QueryWrapper<>();
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public AndonType save(AndonType andonType) {
        return persist(andonType);
    }

    @Override
    public void update(AndonType andonType) {
        if (andonType.getId() == null) throw new IllegalArgumentException("ID is required for update");
        persist(andonType);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private AndonType persist(AndonType entity) {
        AndonTypeDO dto = new AndonTypeDO();
        BeanUtils.copyProperties(entity, dto);
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("AndonType not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private AndonType toEntity(AndonTypeDO dto) {
        AndonType entity = new AndonType();
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }
}