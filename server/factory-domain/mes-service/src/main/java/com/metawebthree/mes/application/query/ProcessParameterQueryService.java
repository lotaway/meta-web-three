package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;

/**
 * 工艺参数配置查询服务
 * 处理工艺参数的查询检索
 */
@Service
@Transactional(readOnly = true)
public class ProcessParameterQueryService {
    
    @Autowired
    private ProcessParameterRepository repository;
    
    /**
     * 根据ID查询
     */
    public ProcessParameter findById(Long id) {
        return repository.findById(id);
    }
    
    /**
     * 根据参数编码查询
     */
    public ProcessParameter findByParamCode(String paramCode) {
        return repository.findByParamCode(paramCode);
    }
    
    /**
     * 根据工艺路线ID查询所有参数
     */
    public List<ProcessParameter> findByRouteId(Long routeId) {
        return repository.findByRouteIdOrderByStepNoAscDisplayOrderAsc(routeId);
    }
    
    /**
     * 根据工艺路线编码查询所有参数
     */
    public List<ProcessParameter> findByRouteCode(String routeCode) {
        return repository.findByRouteCodeOrderByStepNoAscDisplayOrderAsc(routeCode);
    }
    
    /**
     * 根据工序查询参数
     */
    public List<ProcessParameter> findByStep(Long routeId, Integer stepNo) {
        return repository.findByRouteIdAndStepNoOrderByDisplayOrderAsc(routeId, stepNo);
    }
    
    /**
     * 根据参数类型查询
     */
    public List<ProcessParameter> findByParamType(ProcessParameter.ParamType paramType) {
        return repository.findByParamType(paramType);
    }
    
    /**
     * 查询所有激活的参数
     */
    public List<ProcessParameter> findActiveParameters() {
        return repository.findByStatus(ProcessParameter.ParamStatus.ACTIVE);
    }
    
    /**
     * 根据参数分组查询
     */
    public List<ProcessParameter> findByParamGroup(String paramGroup) {
        return repository.findByParamGroup(paramGroup);
    }
    
    /**
     * 根据工艺路线ID统计参数数量
     */
    public long countByRouteId(Long routeId) {
        return repository.countByRouteId(routeId);
    }
    
    /**
     * 根据工序统计参数数量
     */
    public long countByStep(Long routeId, Integer stepNo) {
        return repository.countByRouteIdAndStepNo(routeId, stepNo);
    }
    
    /**
     * 验证参数值是否合格
     */
    public ValidationResult validateValue(Long paramId, BigDecimal value) {
        ProcessParameter param = repository.findById(paramId);
        if (param == null) {
            throw new IllegalArgumentException("参数不存在: " + paramId);
        }
        
        ValidationResult result = new ValidationResult();
        result.setParamId(paramId);
        result.setParamCode(param.getParamCode());
        result.setParamName(param.getParamName());
        result.setInputValue(value);
        result.setStandardValue(param.getStandardValue());
        result.setUpperLimit(param.getUpperLimit());
        result.setLowerLimit(param.getLowerLimit());
        
        if (value == null) {
            if (param.getRequired()) {
                result.setValid(false);
                result.setMessage("参数值不能为空");
                return result;
            }
            result.setValid(true);
            result.setMessage("参数值已通过校验");
            return result;
        }
        
        // 检查是否在规格范围内
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
        
        // 检查是否超差
        if (param.isOutOfTolerance(value)) {
            BigDecimal deviation = param.calculateDeviation(value);
            result.setDeviation(deviation);
            result.setOutOfTolerance(true);
            result.setMessage("参数值超出报警阈值，偏差: " + deviation + "%");
            result.setValid(true); // 超差仍然算通过，但需要标记
            return result;
        }
        
        result.setValid(true);
        result.setMessage("参数值已通过校验");
        return result;
    }
    
    /**
     * 校验结果
     */
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
        
        // Getters and Setters
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