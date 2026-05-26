package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class ProcessParameterCommandService {
    
    private final ProcessParameterRepository repository;
    
    public ProcessParameterCommandService(ProcessParameterRepository repository) {
        this.repository = repository;
    }
    
    public ProcessParameter createParameter(
            String paramCode,
            String paramName,
            Long routeId,
            String routeCode,
            Integer stepNo,
            String stepCode,
            ProcessParameter.ParamType paramType,
            ProcessParameter.DataType dataType,
            String unit,
            BigDecimal standardValue,
            BigDecimal upperLimit,
            BigDecimal lowerLimit,
            ProcessParameter.CollectionMethod collectionMethod,
            String deviceAddress,
            Boolean required,
            String validationRule,
            BigDecimal alarmThreshold,
            Integer displayOrder,
            String paramGroup,
            String remark) {
        
        if (repository.findByParamCode(paramCode).isPresent()) {
            throw new IllegalArgumentException("参数编码已存在: " + paramCode);
        }
        
        ProcessParameter parameter = ProcessParameter.create(paramCode, paramName, routeId, routeCode, stepNo, stepCode, paramType, dataType);
        parameter.setUnit(unit);
        parameter.setStandardValue(standardValue);
        parameter.setUpperLimit(upperLimit);
        parameter.setLowerLimit(lowerLimit);
        parameter.setCollectionMethod(collectionMethod);
        parameter.setDeviceAddress(deviceAddress);
        parameter.setIsRequired(required);
        parameter.setValidationRule(validationRule);
        parameter.setAlarmThreshold(alarmThreshold);
        parameter.setDisplayOrder(displayOrder);
        parameter.setParamGroup(paramGroup);
        parameter.setRemark(remark);
        
        return repository.save(parameter);
    }
    
    public ProcessParameter updateParameter(Long id, ProcessParameter updated) {
        ProcessParameter parameter = repository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("参数不存在: " + id));
        
        if (updated.getParamName() != null) {
            parameter.setParamName(updated.getParamName());
        }
        if (updated.getUnit() != null) {
            parameter.setUnit(updated.getUnit());
        }
        if (updated.getStandardValue() != null) {
            parameter.setStandardValue(updated.getStandardValue());
        }
        if (updated.getUpperLimit() != null) {
            parameter.setUpperLimit(updated.getUpperLimit());
        }
        if (updated.getLowerLimit() != null) {
            parameter.setLowerLimit(updated.getLowerLimit());
        }
        if (updated.getCollectionMethod() != null) {
            parameter.setCollectionMethod(updated.getCollectionMethod());
        }
        if (updated.getDeviceAddress() != null) {
            parameter.setDeviceAddress(updated.getDeviceAddress());
        }
        if (updated.getIsRequired() != null) {
            parameter.setIsRequired(updated.getIsRequired());
        }
        if (updated.getValidationRule() != null) {
            parameter.setValidationRule(updated.getValidationRule());
        }
        if (updated.getAlarmThreshold() != null) {
            parameter.setAlarmThreshold(updated.getAlarmThreshold());
        }
        if (updated.getDisplayOrder() != null) {
            parameter.setDisplayOrder(updated.getDisplayOrder());
        }
        if (updated.getParamGroup() != null) {
            parameter.setParamGroup(updated.getParamGroup());
        }
        if (updated.getRemark() != null) {
            parameter.setRemark(updated.getRemark());
        }
        
        return repository.save(parameter);
    }
    
    public ProcessParameter updateStatus(Long id, ProcessParameter.ParamStatus status) {
        ProcessParameter parameter = repository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("参数不存在: " + id));
        
        parameter.setStatus(status);
        return repository.save(parameter);
    }
    
    public void deleteParameter(Long id) {
        if (repository.findById(id).isEmpty()) {
            throw new IllegalArgumentException("参数不存在: " + id);
        }
        repository.deleteById(id);
    }
    
    public void deleteParameters(List<Long> ids) {
        repository.deleteAllById(ids);
    }
    
    public List<ProcessParameter> copyToRoute(Long sourceRouteId, Long targetRouteId, 
                                               String targetRouteCode, Integer newStepOffset) {
        List<ProcessParameter> sourceParams = repository.findByRouteIdOrderByStepNoAscDisplayOrderAsc(sourceRouteId);
        
        for (ProcessParameter param : sourceParams) {
            param.setId(null);
            param.setRouteId(targetRouteId);
            param.setRouteCode(targetRouteCode);
            if (newStepOffset != null && param.getStepNo() != null) {
                param.setStepNo(param.getStepNo() + newStepOffset);
            }
        }
        
        return repository.saveAll(sourceParams);
    }
}