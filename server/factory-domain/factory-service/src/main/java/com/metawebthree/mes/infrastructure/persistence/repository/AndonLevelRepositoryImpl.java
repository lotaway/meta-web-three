package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.AndonLevel;
import com.metawebthree.mes.domain.repository.AndonLevelRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.AndonLevelDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.AndonLevelMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class AndonLevelRepositoryImpl implements AndonLevelRepository {
    private final AndonLevelMapper mapper;
    private final ObjectMapper json;

    public AndonLevelRepositoryImpl(AndonLevelMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public Optional<AndonLevel> findById(Long id) {
        QueryWrapper<AndonLevelDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<AndonLevel> findByLevelCode(String levelCode) {
        QueryWrapper<AndonLevelDO> query = new QueryWrapper<>();
        query.eq("level_code", levelCode);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<AndonLevel> findByLevelValue(Integer levelValue) {
        QueryWrapper<AndonLevelDO> query = new QueryWrapper<>();
        query.eq("level_value", levelValue);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<AndonLevel> findByStatus(AndonLevel.AndonLevelStatus status) {
        QueryWrapper<AndonLevelDO> query = new QueryWrapper<>();
        query.eq("status", status);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonLevel> findAll() {
        QueryWrapper<AndonLevelDO> query = new QueryWrapper<>();
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public AndonLevel save(AndonLevel andonLevel) {
        return persist(andonLevel);
    }

    @Override
    public void update(AndonLevel andonLevel) {
        if (andonLevel.getId() == null) throw new IllegalArgumentException("ID is required for update");
        persist(andonLevel);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private AndonLevel persist(AndonLevel entity) {
        AndonLevelDO dto = new AndonLevelDO();
        BeanUtils.copyProperties(entity, dto);
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("AndonLevel not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private AndonLevel toEntity(AndonLevelDO dto) {
        AndonLevel entity = new AndonLevel();
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }
}