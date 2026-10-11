package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessParameterDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessParameterMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Repository
public class ProcessParameterRepositoryImpl implements ProcessParameterRepository {
    
    @Autowired
    private ProcessParameterMapper processParameterMapper;
    
    @Override
    public Optional<ProcessParameter> findById(Long id) {
        ProcessParameterDO doObj = processParameterMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<ProcessParameter> findByIds(List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return List.of();
        }
        List<ProcessParameterDO> doList = processParameterMapper.selectBatchIds(ids);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public Optional<ProcessParameter> findByParamCode(String paramCode) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getParamCode, paramCode);
        ProcessParameterDO doObj = processParameterMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }
    
    @Override
    public List<ProcessParameter> findByRouteIdOrderByStepNoAscDisplayOrderAsc(Long routeId) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getRouteId, routeId)
                .orderByAsc(ProcessParameterDO::getStepNo)
                .orderByAsc(ProcessParameterDO::getDisplayOrder);
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByRouteCodeOrderByStepNoAscDisplayOrderAsc(String routeCode) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getRouteCode, routeCode)
                .orderByAsc(ProcessParameterDO::getStepNo)
                .orderByAsc(ProcessParameterDO::getDisplayOrder);
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByRouteIdAndStepNoOrderByDisplayOrderAsc(Long routeId, Integer stepNo) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getRouteId, routeId)
                .eq(ProcessParameterDO::getStepNo, stepNo)
                .orderByAsc(ProcessParameterDO::getDisplayOrder);
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByParamType(ProcessParameter.ParamType paramType) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getParamType, paramType.name());
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByStatus(ProcessParameter.ParamStatus status) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getStatus, status.name());
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByParamGroup(String paramGroup) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getParamGroup, paramGroup);
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public boolean existsByParamCode(String paramCode) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getParamCode, paramCode);
        return processParameterMapper.selectCount(wrapper) > 0;
    }
    
    @Override
    public long countByRouteId(Long routeId) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getRouteId, routeId);
        return processParameterMapper.selectCount(wrapper);
    }
    
    @Override
    public long countByRouteIdAndStepNo(Long routeId, Integer stepNo) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getRouteId, routeId)
               .eq(ProcessParameterDO::getStepNo, stepNo);
        return processParameterMapper.selectCount(wrapper);
    }
    
    @Override
    public List<ProcessParameter> findActiveByRouteId(Long routeId) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getRouteId, routeId)
                .eq(ProcessParameterDO::getStatus, ProcessParameter.ParamStatus.ACTIVE.name())
                .orderByAsc(ProcessParameterDO::getStepNo)
                .orderByAsc(ProcessParameterDO::getDisplayOrder);
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByParamTypeAndStatus(ProcessParameter.ParamType paramType, ProcessParameter.ParamStatus status) {
        LambdaQueryWrapper<ProcessParameterDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessParameterDO::getParamType, paramType.name())
                .eq(ProcessParameterDO::getStatus, status.name());
        List<ProcessParameterDO> doList = processParameterMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public ProcessParameter save(ProcessParameter parameter) {
        ProcessParameterDO doObj = toDO(parameter);
        if (parameter.getId() == null) {
            processParameterMapper.insert(doObj);
            parameter.setId(doObj.getId());
        } else {
            processParameterMapper.updateById(doObj);
        }
        return parameter;
    }
    
    @Override
    public List<ProcessParameter> saveAll(List<ProcessParameter> parameters) {
        List<ProcessParameter> saved = new ArrayList<>();
        for (ProcessParameter parameter : parameters) {
            saved.add(save(parameter));
        }
        return saved;
    }
    
    @Override
    public void deleteById(Long id) {
        processParameterMapper.deleteById(id);
    }
    
    @Override
    public void deleteAllById(List<Long> ids) {
        processParameterMapper.deleteBatchIds(ids);
    }
    
    @Override
    public boolean existsById(Long id) {
        return processParameterMapper.selectById(id) != null;
    }
    
    private ProcessParameter toEntity(ProcessParameterDO doObj) {
        if (doObj == null) {
            return null;
        }
        ProcessParameter entity = new ProcessParameter();
        entity.setId(doObj.getId());
        entity.setParamCode(doObj.getParamCode());
        entity.setParamName(doObj.getParamName());
        entity.setRouteId(doObj.getRouteId());
        entity.setRouteCode(doObj.getRouteCode());
        entity.setStepNo(doObj.getStepNo());
        entity.setStepCode(doObj.getStepCode());
        entity.setParamType(doObj.getParamType() != null ? ProcessParameter.ParamType.valueOf(doObj.getParamType()) : null);
        entity.setDataType(doObj.getDataType() != null ? ProcessParameter.DataType.valueOf(doObj.getDataType()) : null);
        entity.setUnit(doObj.getUnit());
        entity.setStandardValue(doObj.getStandardValue());
        entity.setUpperLimit(doObj.getUpperLimit());
        entity.setLowerLimit(doObj.getLowerLimit());
        entity.setCollectionMethod(doObj.getCollectionMethod() != null ? ProcessParameter.CollectionMethod.valueOf(doObj.getCollectionMethod()) : null);
        entity.setDeviceAddress(doObj.getDeviceAddress());
        entity.setIsRequired(doObj.getIsRequired());
        entity.setValidationRule(doObj.getValidationRule());
        entity.setAlarmThreshold(doObj.getAlarmThreshold());
        entity.setDisplayOrder(doObj.getDisplayOrder());
        entity.setParamGroup(doObj.getParamGroup());
        entity.setRemark(doObj.getRemark());
        entity.setStatus(doObj.getStatus() != null ? ProcessParameter.ParamStatus.valueOf(doObj.getStatus()) : null);
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private ProcessParameterDO toDO(ProcessParameter entity) {
        if (entity == null) {
            return null;
        }
        ProcessParameterDO doObj = new ProcessParameterDO();
        doObj.setId(entity.getId());
        doObj.setParamCode(entity.getParamCode());
        doObj.setParamName(entity.getParamName());
        doObj.setRouteId(entity.getRouteId());
        doObj.setRouteCode(entity.getRouteCode());
        doObj.setStepNo(entity.getStepNo());
        doObj.setStepCode(entity.getStepCode());
        doObj.setParamType(entity.getParamType() != null ? entity.getParamType().name() : null);
        doObj.setDataType(entity.getDataType() != null ? entity.getDataType().name() : null);
        doObj.setUnit(entity.getUnit());
        doObj.setStandardValue(entity.getStandardValue());
        doObj.setUpperLimit(entity.getUpperLimit());
        doObj.setLowerLimit(entity.getLowerLimit());
        doObj.setCollectionMethod(entity.getCollectionMethod() != null ? entity.getCollectionMethod().name() : null);
        doObj.setDeviceAddress(entity.getDeviceAddress());
        doObj.setIsRequired(entity.getIsRequired());
        doObj.setValidationRule(entity.getValidationRule());
        doObj.setAlarmThreshold(entity.getAlarmThreshold());
        doObj.setDisplayOrder(entity.getDisplayOrder());
        doObj.setParamGroup(entity.getParamGroup());
        doObj.setRemark(entity.getRemark());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        return doObj;
    }
}