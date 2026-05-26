package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.mes.application.command.ProcessParameterCommandService;
import com.metawebthree.mes.application.query.ProcessParameterQueryService;
import com.metawebthree.mes.domain.entity.ProcessParameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 工艺参数配置控制器
 * 提供工艺参数的 REST API
 */
@RestController
@RequestMapping("/api/mes/process-parameters")
public class ProcessParameterController {
    
    @Autowired
    private ProcessParameterCommandService commandService;
    
    @Autowired
    private ProcessParameterQueryService queryService;
    
    /**
     * 创建工艺参数
     */
    @PostMapping
    public ResponseEntity<ProcessParameter> create(@RequestBody ProcessParameterRequest request) {
        ProcessParameter param = commandService.createParameter(
                request.getParamCode(),
                request.getParamName(),
                request.getRouteId(),
                request.getRouteCode(),
                request.getStepNo(),
                request.getStepCode(),
                request.getParamType(),
                request.getDataType(),
                request.getUnit(),
                request.getStandardValue(),
                request.getUpperLimit(),
                request.getLowerLimit(),
                request.getCollectionMethod(),
                request.getDeviceAddress(),
                request.getRequired(),
                request.getValidationRule(),
                request.getAlarmThreshold(),
                request.getDisplayOrder(),
                request.getParamGroup(),
                request.getRemark()
        );
        return ResponseEntity.status(HttpStatus.CREATED).body(param);
    }
    
    /**
     * 更新工艺参数
     */
    @PutMapping("/{id}")
    public ResponseEntity<ProcessParameter> update(@PathVariable Long id, @RequestBody ProcessParameter updated) {
        ProcessParameter param = commandService.updateParameter(id, updated);
        return ResponseEntity.ok(param);
    }
    
    /**
     * 更新参数状态
     */
    @PatchMapping("/{id}/status")
    public ResponseEntity<ProcessParameter> updateStatus(
            @PathVariable Long id,
            @RequestParam ProcessParameter.ParamStatus status) {
        ProcessParameter param = commandService.updateStatus(id, status);
        return ResponseEntity.ok(param);
    }
    
    /**
     * 删除工艺参数
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        commandService.deleteParameter(id);
        return ResponseEntity.noContent().build();
    }
    
    /**
     * 批量删除工艺参数
     */
    @DeleteMapping("/batch")
    public ResponseEntity<Void> deleteBatch(@RequestBody List<Long> ids) {
        commandService.deleteParameters(ids);
        return ResponseEntity.noContent().build();
    }
    
    /**
     * 根据ID查询
     */
    @GetMapping("/{id}")
    public ResponseEntity<ProcessParameter> getById(@PathVariable Long id) {
        ProcessParameter param = queryService.findById(id);
        if (param == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(param);
    }
    
    /**
     * 根据参数编码查询
     */
    @GetMapping("/code/{paramCode}")
    public ResponseEntity<ProcessParameter> getByParamCode(@PathVariable String paramCode) {
        ProcessParameter param = queryService.findByParamCode(paramCode);
        if (param == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(param);
    }
    
    /**
     * 根据工艺路线ID查询
     */
    @GetMapping("/route/{routeId}")
    public ResponseEntity<List<ProcessParameter>> getByRouteId(@PathVariable Long routeId) {
        List<ProcessParameter> params = queryService.findByRouteId(routeId);
        return ResponseEntity.ok(params);
    }
    
    /**
     * 根据工艺路线编码查询
     */
    @GetMapping("/route-code/{routeCode}")
    public ResponseEntity<List<ProcessParameter>> getByRouteCode(@PathVariable String routeCode) {
        List<ProcessParameter> params = queryService.findByRouteCode(routeCode);
        return ResponseEntity.ok(params);
    }
    
    /**
     * 根据工序查询
     */
    @GetMapping("/step/{routeId}/{stepNo}")
    public ResponseEntity<List<ProcessParameter>> getByStep(
            @PathVariable Long routeId,
            @PathVariable Integer stepNo) {
        List<ProcessParameter> params = queryService.findByStep(routeId, stepNo);
        return ResponseEntity.ok(params);
    }
    
    /**
     * 根据参数类型查询
     */
    @GetMapping("/type/{paramType}")
    public ResponseEntity<List<ProcessParameter>> getByParamType(
            @PathVariable ProcessParameter.ParamType paramType) {
        List<ProcessParameter> params = queryService.findByParamType(paramType);
        return ResponseEntity.ok(params);
    }
    
    /**
     * 查询所有激活的参数
     */
    @GetMapping("/active")
    public ResponseEntity<List<ProcessParameter>> getActiveParameters() {
        List<ProcessParameter> params = queryService.findActiveParameters();
        return ResponseEntity.ok(params);
    }
    
    /**
     * 根据参数分组查询
     */
    @GetMapping("/group/{paramGroup}")
    public ResponseEntity<List<ProcessParameter>> getByParamGroup(@PathVariable String paramGroup) {
        List<ProcessParameter> params = queryService.findByParamGroup(paramGroup);
        return ResponseEntity.ok(params);
    }
    
    /**
     * 统计工艺路线的参数数量
     */
    @GetMapping("/route/{routeId}/count")
    public ResponseEntity<Map<String, Long>> countByRouteId(@PathVariable Long routeId) {
        long count = queryService.countByRouteId(routeId);
        Map<String, Long> result = new HashMap<>();
        result.put("count", count);
        return ResponseEntity.ok(result);
    }
    
    /**
     * 验证参数值
     */
    @PostMapping("/{id}/validate")
    public ResponseEntity<ProcessParameterQueryService.ValidationResult> validateValue(
            @PathVariable Long id,
            @RequestParam BigDecimal value) {
        ProcessParameterQueryService.ValidationResult result = queryService.validateValue(id, value);
        return ResponseEntity.ok(result);
    }
    
    /**
     * 批量验证参数值
     */
    @PostMapping("/validate-batch")
    public ResponseEntity<List<ProcessParameterQueryService.ValidationResult>> validateBatch(
            @RequestBody List<ValidateRequest> requests) {
        List<ProcessParameterQueryService.ValidationResult> results = requests.stream()
                .map(req -> queryService.validateValue(req.getParamId(), req.getValue()))
                .toList();
        return ResponseEntity.ok(results);
    }
    
    /**
     * 复制工艺参数到新工艺路线
     */
    @PostMapping("/copy")
    public ResponseEntity<List<ProcessParameter>> copyToRoute(@RequestBody CopyRequest request) {
        List<ProcessParameter> params = commandService.copyToRoute(
                request.getSourceRouteId(),
                request.getTargetRouteId(),
                request.getTargetRouteCode(),
                request.getStepOffset()
        );
        return ResponseEntity.status(HttpStatus.CREATED).body(params);
    }
    
    /**
     * 请求对象
     */
    public static class ProcessParameterRequest {
        private String paramCode;
        private String paramName;
        private Long routeId;
        private String routeCode;
        private Integer stepNo;
        private String stepCode;
        private ProcessParameter.ParamType paramType;
        private ProcessParameter.DataType dataType;
        private String unit;
        private BigDecimal standardValue;
        private BigDecimal upperLimit;
        private BigDecimal lowerLimit;
        private ProcessParameter.CollectionMethod collectionMethod;
        private String deviceAddress;
        private Boolean required;
        private String validationRule;
        private BigDecimal alarmThreshold;
        private Integer displayOrder;
        private String paramGroup;
        private String remark;
        
        // Getters and Setters
        public String getParamCode() { return paramCode; }
        public void setParamCode(String paramCode) { this.paramCode = paramCode; }
        public String getParamName() { return paramName; }
        public void setParamName(String paramName) { this.paramName = paramName; }
        public Long getRouteId() { return routeId; }
        public void setRouteId(Long routeId) { this.routeId = routeId; }
        public String getRouteCode() { return routeCode; }
        public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
        public Integer getStepNo() { return stepNo; }
        public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
        public String getStepCode() { return stepCode; }
        public void setStepCode(String stepCode) { this.stepCode = stepCode; }
        public ProcessParameter.ParamType getParamType() { return paramType; }
        public void setParamType(ProcessParameter.ParamType paramType) { this.paramType = paramType; }
        public ProcessParameter.DataType getDataType() { return dataType; }
        public void setDataType(ProcessParameter.DataType dataType) { this.dataType = dataType; }
        public String getUnit() { return unit; }
        public void setUnit(String unit) { this.unit = unit; }
        public BigDecimal getStandardValue() { return standardValue; }
        public void setStandardValue(BigDecimal standardValue) { this.standardValue = standardValue; }
        public BigDecimal getUpperLimit() { return upperLimit; }
        public void setUpperLimit(BigDecimal upperLimit) { this.upperLimit = upperLimit; }
        public BigDecimal getLowerLimit() { return lowerLimit; }
        public void setLowerLimit(BigDecimal lowerLimit) { this.lowerLimit = lowerLimit; }
        public ProcessParameter.CollectionMethod getCollectionMethod() { return collectionMethod; }
        public void setCollectionMethod(ProcessParameter.CollectionMethod collectionMethod) { this.collectionMethod = collectionMethod; }
        public String getDeviceAddress() { return deviceAddress; }
        public void setDeviceAddress(String deviceAddress) { this.deviceAddress = deviceAddress; }
        public Boolean getRequired() { return required; }
        public void setRequired(Boolean required) { this.required = required; }
        public String getValidationRule() { return validationRule; }
        public void setValidationRule(String validationRule) { this.validationRule = validationRule; }
        public BigDecimal getAlarmThreshold() { return alarmThreshold; }
        public void setAlarmThreshold(BigDecimal alarmThreshold) { this.alarmThreshold = alarmThreshold; }
        public Integer getDisplayOrder() { return displayOrder; }
        public void setDisplayOrder(Integer displayOrder) { this.displayOrder = displayOrder; }
        public String getParamGroup() { return paramGroup; }
        public void setParamGroup(String paramGroup) { this.paramGroup = paramGroup; }
        public String getRemark() { return remark; }
        public void setRemark(String remark) { this.remark = remark; }
    }
    
    /**
     * 校验请求
     */
    public static class ValidateRequest {
        private Long paramId;
        private BigDecimal value;
        
        public Long getParamId() { return paramId; }
        public void setParamId(Long paramId) { this.paramId = paramId; }
        public BigDecimal getValue() { return value; }
        public void setValue(BigDecimal value) { this.value = value; }
    }
    
    /**
     * 复制请求
     */
    public static class CopyRequest {
        private Long sourceRouteId;
        private Long targetRouteId;
        private String targetRouteCode;
        private Integer stepOffset;
        
        public Long getSourceRouteId() { return sourceRouteId; }
        public void setSourceRouteId(Long sourceRouteId) { this.sourceRouteId = sourceRouteId; }
        public Long getTargetRouteId() { return targetRouteId; }
        public void setTargetRouteId(Long targetRouteId) { this.targetRouteId = targetRouteId; }
        public String getTargetRouteCode() { return targetRouteCode; }
        public void setTargetRouteCode(String targetRouteCode) { this.targetRouteCode = targetRouteCode; }
        public Integer getStepOffset() { return stepOffset; }
        public void setStepOffset(Integer stepOffset) { this.stepOffset = stepOffset; }
    }
}