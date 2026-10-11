package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        
        Map<String, Object> updates = new HashMap<>();
        updates.put("paramName", updated.getParamName());
        updates.put("unit", updated.getUnit());
        updates.put("standardValue", updated.getStandardValue());
        updates.put("upperLimit", updated.getUpperLimit());
        updates.put("lowerLimit", updated.getLowerLimit());
        updates.put("collectionMethod", updated.getCollectionMethod());
        updates.put("deviceAddress", updated.getDeviceAddress());
        updates.put("isRequired", updated.getIsRequired());
        updates.put("validationRule", updated.getValidationRule());
        updates.put("alarmThreshold", updated.getAlarmThreshold());
        updates.put("displayOrder", updated.getDisplayOrder());
        updates.put("paramGroup", updated.getParamGroup());
        updates.put("remark", updated.getRemark());
        
        applyUpdates(parameter, updates);
        
        return repository.save(parameter);
    }
    
    private void applyUpdates(ProcessParameter parameter, Map<String, Object> updates) {
        if (updates.get("paramName") != null) parameter.setParamName((String) updates.get("paramName"));
        if (updates.get("unit") != null) parameter.setUnit((String) updates.get("unit"));
        if (updates.get("standardValue") != null) parameter.setStandardValue((BigDecimal) updates.get("standardValue"));
        if (updates.get("upperLimit") != null) parameter.setUpperLimit((BigDecimal) updates.get("upperLimit"));
        if (updates.get("lowerLimit") != null) parameter.setLowerLimit((BigDecimal) updates.get("lowerLimit"));
        if (updates.get("collectionMethod") != null) parameter.setCollectionMethod((ProcessParameter.CollectionMethod) updates.get("collectionMethod"));
        if (updates.get("deviceAddress") != null) parameter.setDeviceAddress((String) updates.get("deviceAddress"));
        if (updates.get("isRequired") != null) parameter.setIsRequired((Boolean) updates.get("isRequired"));
        if (updates.get("validationRule") != null) parameter.setValidationRule((String) updates.get("validationRule"));
        if (updates.get("alarmThreshold") != null) parameter.setAlarmThreshold((BigDecimal) updates.get("alarmThreshold"));
        if (updates.get("displayOrder") != null) parameter.setDisplayOrder((Integer) updates.get("displayOrder"));
        if (updates.get("paramGroup") != null) parameter.setParamGroup((String) updates.get("paramGroup"));
        if (updates.get("remark") != null) parameter.setRemark((String) updates.get("remark"));
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
        
        List<ProcessParameter> newParams = new ArrayList<>();
        for (ProcessParameter param : sourceParams) {
            // 计算新的 stepNo
            Integer newStepNo = param.getStepNo();
            if (newStepOffset != null && param.getStepNo() != null) {
                newStepNo = param.getStepNo() + newStepOffset;
            }
            
            // 创建新实体，避免修改原实体（修复数据破坏BUG）
            ProcessParameter newParam = ProcessParameter.create(
                param.getParamCode(),
                param.getParamName(),
                targetRouteId,
                targetRouteCode,
                newStepNo,
                param.getStepCode(),
                param.getParamType(),
                param.getDataType()
            );
            newParam.setUnit(param.getUnit());
            newParam.setStandardValue(param.getStandardValue());
            newParam.setUpperLimit(param.getUpperLimit());
            newParam.setLowerLimit(param.getLowerLimit());
            newParam.setCollectionMethod(param.getCollectionMethod());
            newParam.setDeviceAddress(param.getDeviceAddress());
            newParam.setIsRequired(param.getIsRequired());
            newParam.setValidationRule(param.getValidationRule());
            newParam.setAlarmThreshold(param.getAlarmThreshold());
            newParam.setDisplayOrder(param.getDisplayOrder());
            newParam.setParamGroup(param.getParamGroup());
            newParam.setRemark(param.getRemark());
            
            newParams.add(newParam);
        }
        
        return repository.saveAll(newParams);
    }
}