package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition.DispositionStep;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition.DispositionType;
import com.metawebthree.mes.domain.repository.NonConformanceDispositionRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.NonConformanceDispositionDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.NonConformanceDispositionMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class NonConformanceDispositionRepositoryImpl implements NonConformanceDispositionRepository {
    
    @Autowired
    private NonConformanceDispositionMapper mapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<NonConformanceDisposition> findById(Long id) {
        NonConformanceDispositionDO doObj = mapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public Optional<NonConformanceDisposition> findByDispositionCode(String dispositionCode) {
        LambdaQueryWrapper<NonConformanceDispositionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(NonConformanceDispositionDO::getDispositionCode, dispositionCode);
        NonConformanceDispositionDO doObj = mapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<NonConformanceDisposition> findAll() {
        List<NonConformanceDispositionDO> doList = mapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<NonConformanceDisposition> findByType(DispositionType type) {
        LambdaQueryWrapper<NonConformanceDispositionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(NonConformanceDispositionDO::getType, type.name());
        List<NonConformanceDispositionDO> doList = mapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<NonConformanceDisposition> findByIsEnabled(Boolean isEnabled) {
        LambdaQueryWrapper<NonConformanceDispositionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(NonConformanceDispositionDO::getIsEnabled, isEnabled);
        List<NonConformanceDispositionDO> doList = mapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public NonConformanceDisposition save(NonConformanceDisposition disposition) {
        NonConformanceDispositionDO doObj = toDO(disposition);
        if (disposition.getId() == null) {
            mapper.insert(doObj);
            disposition.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return disposition;
    }
    
    @Override
    public void update(NonConformanceDisposition disposition) {
        if (disposition.getId() != null) {
            NonConformanceDispositionDO doObj = toDO(disposition);
            mapper.updateById(doObj);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
    
    @Override
    public Boolean existsByDispositionCode(String dispositionCode) {
        LambdaQueryWrapper<NonConformanceDispositionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(NonConformanceDispositionDO::getDispositionCode, dispositionCode);
        return mapper.selectCount(wrapper) > 0;
    }
    
    private NonConformanceDisposition toEntity(NonConformanceDispositionDO doObj) {
        if (doObj == null) {
            return null;
        }
        NonConformanceDisposition entity = new NonConformanceDisposition();
        entity.setId(doObj.getId());
        entity.setDispositionCode(doObj.getDispositionCode());
        entity.setDispositionName(doObj.getDispositionName());
        if (doObj.getType() != null) {
            entity.setType(DispositionType.valueOf(doObj.getType()));
        }
        entity.setIsEnabled(doObj.getIsEnabled());
        entity.setSortOrder(doObj.getSortOrder());
        if (doObj.getStepsJson() != null) {
            try {
                List<DispositionStep> steps = objectMapper.readValue(doObj.getStepsJson(), 
                    new TypeReference<List<DispositionStep>>() {});
                entity.setSteps(steps);
            } catch (JsonProcessingException e) {
                entity.setSteps(List.of());
            }
        }
        return entity;
    }
    
    private NonConformanceDispositionDO toDO(NonConformanceDisposition entity) {
        if (entity == null) {
            return null;
        }
        NonConformanceDispositionDO doObj = new NonConformanceDispositionDO();
        doObj.setId(entity.getId());
        doObj.setDispositionCode(entity.getDispositionCode());
        doObj.setDispositionName(entity.getDispositionName());
        doObj.setType(entity.getType() != null ? entity.getType().name() : null);
        doObj.setIsEnabled(entity.getIsEnabled());
        doObj.setSortOrder(entity.getSortOrder());
        if (entity.getSteps() != null) {
            try {
                doObj.setStepsJson(objectMapper.writeValueAsString(entity.getSteps()));
            } catch (JsonProcessingException e) {
                doObj.setStepsJson("[]");
            }
        }
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}