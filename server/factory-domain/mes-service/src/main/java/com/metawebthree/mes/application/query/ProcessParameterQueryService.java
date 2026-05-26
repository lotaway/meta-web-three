package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;

@Service
@Transactional(readOnly = true)
public class ProcessParameterQueryService {
    
    private final ProcessParameterRepository repository;
    
    public ProcessParameterQueryService(ProcessParameterRepository repository) {
        this.repository = repository;
    }
    
    public Optional<ProcessParameter> findById(Long id) {
        return repository.findById(id);
    }
    
    public Optional<ProcessParameter> findByParamCode(String paramCode) {
        return repository.findByParamCode(paramCode);
    }
    
    public List<ProcessParameter> findByRouteId(Long routeId) {
        return repository.findByRouteIdOrderByStepNoAscDisplayOrderAsc(routeId);
    }
    
    public List<ProcessParameter> findByRouteCode(String routeCode) {
        return repository.findByRouteCodeOrderByStepNoAscDisplayOrderAsc(routeCode);
    }
    
    public List<ProcessParameter> findByStep(Long routeId, Integer stepNo) {
        return repository.findByRouteIdAndStepNoOrderByDisplayOrderAsc(routeId, stepNo);
    }
    
    public List<ProcessParameter> findByParamType(ProcessParameter.ParamType paramType) {
        return repository.findByParamType(paramType);
    }
    
    public List<ProcessParameter> findActiveParameters() {
        return repository.findByStatus(ProcessParameter.ParamStatus.ACTIVE);
    }
    
    public List<ProcessParameter> findByParamGroup(String paramGroup) {
        return repository.findByParamGroup(paramGroup);
    }
    
    public long countByRouteId(Long routeId) {
        return repository.countByRouteId(routeId);
    }
    
    public long countByStep(Long routeId, Integer stepNo) {
        return repository.countByRouteIdAndStepNo(routeId, stepNo);
    }
    
    public ValidationResult validateValue(Long paramId, BigDecimal value) {
        ProcessParameter param = repository.findById(paramId)
                .orElseThrow(() -> new IllegalArgumentException("参数不存在: " + paramId));
        
        ValidationResult result = new ValidationResult();
        result.setParamId(paramId);
        result.setParamCode(param.getParamCode());
        result.setParamName(param.getParamName());
        result.setInputValue(value);
        result.setStandardValue(param.getStandardValue());
        result.setUpperLimit(param.getUpperLimit());
        result.setLowerLimit(param.getLowerLimit());
        
        if (value == null) {
            if (param.getIsRequired()) {
                result.setValid(false);
                result.setMessage("参数值不能为空");
                return result;
            }
            result.setValid(true);
            result.setMessage("参数值已通过校验");
            return result;
        }
        
        boolean inRange = param.validateValue(value);
        if (!inRange) {
            result.setValid(false);
            if (param.getUpperLimit() != null && value.compareTo(param.getUpperLimit()) > 0) {
                result.setMessage("参数值超过上限: " + param.getUpperLimit());
            } else if (param.getLowerLimit() != null && value.compareTo(param.getLowerLimit()) < 0) {
                result.setMessage("参数值低于下限: " + param.getLowerLimit());
            } else {
                result.setMessage("参数值校验失败");
            }
            return result;
        }
        
        if (param.isOutOfTolerance(value)) {
            BigDecimal deviation = param.calculateDeviation(value);
            result.setDeviation(deviation);
            result.setOutOfTolerance(true);
            result.setMessage("参数值超出报警阈值，偏差: " + deviation + "%");
            result.setValid(true);
            return result;
        }
        
        result.setValid(true);
        result.setMessage("参数值已通过校验");
        return result;
    }
    
    public static class ValidationResult {
        private Long paramId;
        private String paramCode;
        private String paramName;
        private BigDecimal inputValue;
        private BigDecimal standardValue;
        private BigDecimal upperLimit;
        private BigDecimal lowerLimit;
        private BigDecimal deviation;
        private boolean valid;
        private boolean outOfTolerance;
        private String message;
        
        public Long getParamId() { return paramId; }
        public void setParamId(Long paramId) { this.paramId = paramId; }
        public String getParamCode() { return paramCode; }
        public void setParamCode(String paramCode) { this.paramCode = paramCode; }
        public String getParamName() { return paramName; }
        public void setParamName(String paramName) { this.paramName = paramName; }
        public BigDecimal getInputValue() { return inputValue; }
        public void setInputValue(BigDecimal inputValue) { this.inputValue = inputValue; }
        public BigDecimal getStandardValue() { return standardValue; }
        public void setStandardValue(BigDecimal standardValue) { this.standardValue = standardValue; }
        public BigDecimal getUpperLimit() { return upperLimit; }
        public void setUpperLimit(BigDecimal upperLimit) { this.upperLimit = upperLimit; }
        public BigDecimal getLowerLimit() { return lowerLimit; }
        public void setLowerLimit(BigDecimal lowerLimit) { this.lowerLimit = lowerLimit; }
        public BigDecimal getDeviation() { return deviation; }
        public void setDeviation(BigDecimal deviation) { this.deviation = deviation; }
        public boolean isValid() { return valid; }
        public void setValid(boolean valid) { this.valid = valid; }
        public boolean isOutOfTolerance() { return outOfTolerance; }
        public void setOutOfTolerance(boolean outOfTolerance) { this.outOfTolerance = outOfTolerance; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }
}