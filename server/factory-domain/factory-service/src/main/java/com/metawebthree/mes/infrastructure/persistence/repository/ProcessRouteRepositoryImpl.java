package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.entity.ProcessRoute.ProcessStep;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessRouteDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessRouteMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class ProcessRouteRepositoryImpl implements ProcessRouteRepository {
    
    @Autowired
    private ProcessRouteMapper processRouteMapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<ProcessRoute> findById(Long id) {
        ProcessRouteDO processRouteDO = processRouteMapper.selectById(id);
        return Optional.ofNullable(processRouteDO).map(this::toEntity);
    }
    
    @Override
    public Optional<ProcessRoute> findByRouteCode(String routeCode) {
        LambdaQueryWrapper<ProcessRouteDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessRouteDO::getRouteCode, routeCode);
        ProcessRouteDO processRouteDO = processRouteMapper.selectOne(wrapper);
        return Optional.ofNullable(processRouteDO).map(this::toEntity);
    }
    
    @Override
    public List<ProcessRoute> findByProductCode(String productCode) {
        LambdaQueryWrapper<ProcessRouteDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessRouteDO::getProductCode, productCode);
        List<ProcessRouteDO> doList = processRouteMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessRoute> findByStatus(ProcessRoute.RouteStatus status) {
        LambdaQueryWrapper<ProcessRouteDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessRouteDO::getStatus, status.name());
        List<ProcessRouteDO> doList = processRouteMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public ProcessRoute save(ProcessRoute route) {
        ProcessRouteDO processRouteDO = toDO(route);
        if (route.getId() == null) {
            processRouteMapper.insert(processRouteDO);
            route.setId(processRouteDO.getId());
        } else {
            processRouteMapper.updateById(processRouteDO);
        }
        return route;
    }
    
    @Override
    public void update(ProcessRoute route) {
        if (route.getId() != null) {
            ProcessRouteDO processRouteDO = toDO(route);
            processRouteMapper.updateById(processRouteDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        processRouteMapper.deleteById(id);
    }
    
    private ProcessRoute toEntity(ProcessRouteDO doObj) {
        if (doObj == null) {
            return null;
        }
        ProcessRoute entity = new ProcessRoute();
        entity.setId(doObj.getId());
        entity.setRouteCode(doObj.getRouteCode());
        entity.setRouteName(doObj.getRouteName());
        entity.setProductCode(doObj.getProductCode());
        entity.setVersion(doObj.getVersion());
        entity.setStatus(ProcessRoute.RouteStatus.valueOf(doObj.getStatus()));
        
        if (doObj.getSteps() != null && !doObj.getSteps().isEmpty()) {
            try {
                List<ProcessStep> steps = objectMapper.readValue(doObj.getSteps(), 
                    new TypeReference<List<ProcessStep>>() {});
                entity.setSteps(steps);
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Failed to parse process steps", e);
            }
        }
        
        return entity;
    }
    
    private ProcessRouteDO toDO(ProcessRoute entity) {
        if (entity == null) {
            return null;
        }
        ProcessRouteDO doObj = new ProcessRouteDO();
        doObj.setId(entity.getId());
        doObj.setRouteCode(entity.getRouteCode());
        doObj.setRouteName(entity.getRouteName());
        doObj.setProductCode(entity.getProductCode());
        doObj.setVersion(entity.getVersion());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        
        if (entity.getSteps() != null) {
            try {
                doObj.setSteps(objectMapper.writeValueAsString(entity.getSteps()));
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Failed to serialize process steps", e);
            }
        }
        
        return doObj;
    }
}