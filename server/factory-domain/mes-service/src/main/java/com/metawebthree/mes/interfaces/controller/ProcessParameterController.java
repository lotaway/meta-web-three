package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.mes.application.command.ProcessParameterCommandService;
import com.metawebthree.mes.application.query.ProcessParameterQueryService;
import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.interfaces.dto.ProcessParameterDTO;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/process-parameters")
public class ProcessParameterController {
    
    private final ProcessParameterCommandService commandService;
    private final ProcessParameterQueryService queryService;
    
    public ProcessParameterController(
            ProcessParameterCommandService commandService,
            ProcessParameterQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }
    
    @PostMapping
    public ResponseEntity<ProcessParameterDTO> create(@RequestBody ProcessParameterRequest request) {
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
        return ResponseEntity.status(HttpStatus.CREATED).body(ProcessParameterDTO.fromEntity(param));
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<ProcessParameterDTO> update(@PathVariable Long id, @RequestBody ProcessParameterRequest request) {
        ProcessParameter existing = queryService.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("参数不存在: " + id));
        
        ProcessParameter updated = new ProcessParameter();
        updated.setParamCode(existing.getParamCode());
        updated.setParamName(request.getParamName());
        updated.setRouteId(request.getRouteId());
        updated.setRouteCode(request.getRouteCode());
        updated.setStepNo(request.getStepNo());
        updated.setStepCode(request.getStepCode());
        updated.setParamType(request.getParamType());
        updated.setDataType(request.getDataType());
        updated.setUnit(request.getUnit());
        updated.setStandardValue(request.getStandardValue());
        updated.setUpperLimit(request.getUpperLimit());
        updated.setLowerLimit(request.getLowerLimit());
        updated.setCollectionMethod(request.getCollectionMethod());
        updated.setDeviceAddress(request.getDeviceAddress());
        updated.setIsRequired(request.getRequired());
        updated.setValidationRule(request.getValidationRule());
        updated.setAlarmThreshold(request.getAlarmThreshold());
        updated.setDisplayOrder(request.getDisplayOrder());
        updated.setParamGroup(request.getParamGroup());
        updated.setRemark(request.getRemark());
        
        ProcessParameter param = commandService.updateParameter(id, updated);
        return ResponseEntity.ok(ProcessParameterDTO.fromEntity(param));
    }
    
    @PatchMapping("/{id}/status")
    public ResponseEntity<ProcessParameterDTO> updateStatus(
            @PathVariable Long id,
            @RequestParam ProcessParameter.ParamStatus status) {
        ProcessParameter param = commandService.updateStatus(id, status);
        return ResponseEntity.ok(ProcessParameterDTO.fromEntity(param));
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        commandService.deleteParameter(id);
        return ResponseEntity.noContent().build();
    }
    
    @DeleteMapping("/batch")
    public ResponseEntity<Void> deleteBatch(@RequestBody List<Long> ids) {
        commandService.deleteParameters(ids);
        return ResponseEntity.noContent().build();
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<ProcessParameterDTO> getById(@PathVariable Long id) {
        return queryService.findById(id)
                .map(param -> ResponseEntity.ok(ProcessParameterDTO.fromEntity(param)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{paramCode}")
    public ResponseEntity<ProcessParameterDTO> getByParamCode(@PathVariable String paramCode) {
        return queryService.findByParamCode(paramCode)
                .map(param -> ResponseEntity.ok(ProcessParameterDTO.fromEntity(param)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/route/{routeId}")
    public ResponseEntity<List<ProcessParameterDTO>> getByRouteId(@PathVariable Long routeId) {
        List<ProcessParameterDTO> params = queryService.findByRouteId(routeId).stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(params);
    }
    
    @GetMapping("/route-code/{routeCode}")
    public ResponseEntity<List<ProcessParameterDTO>> getByRouteCode(@PathVariable String routeCode) {
        List<ProcessParameterDTO> params = queryService.findByRouteCode(routeCode).stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(params);
    }
    
    @GetMapping("/step/{routeId}/{stepNo}")
    public ResponseEntity<List<ProcessParameterDTO>> getByStep(
            @PathVariable Long routeId,
            @PathVariable Integer stepNo) {
        List<ProcessParameterDTO> params = queryService.findByStep(routeId, stepNo).stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(params);
    }
    
    @GetMapping("/type/{paramType}")
    public ResponseEntity<List<ProcessParameterDTO>> getByParamType(
            @PathVariable ProcessParameter.ParamType paramType) {
        List<ProcessParameterDTO> params = queryService.findByParamType(paramType).stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(params);
    }
    
    @GetMapping("/active")
    public ResponseEntity<List<ProcessParameterDTO>> getActiveParameters() {
        List<ProcessParameterDTO> params = queryService.findActiveParameters().stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(params);
    }
    
    @GetMapping("/group/{paramGroup}")
    public ResponseEntity<List<ProcessParameterDTO>> getByParamGroup(@PathVariable String paramGroup) {
        List<ProcessParameterDTO> params = queryService.findByParamGroup(paramGroup).stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(params);
    }
    
    @GetMapping("/route/{routeId}/count")
    public ResponseEntity<Map<String, Long>> countByRouteId(@PathVariable Long routeId) {
        long count = queryService.countByRouteId(routeId);
        Map<String, Long> result = new HashMap<>();
        result.put("count", count);
        return ResponseEntity.ok(result);
    }
    
    @PostMapping("/{id}/validate")
    public ResponseEntity<ProcessParameterQueryService.ValidationResult> validateValue(
            @PathVariable Long id,
            @RequestParam BigDecimal value) {
        ProcessParameterQueryService.ValidationResult result = queryService.validateValue(id, value);
        return ResponseEntity.ok(result);
    }
    
    @PostMapping("/validate-batch")
    public ResponseEntity<List<ProcessParameterQueryService.ValidationResult>> validateBatch(
            @RequestBody List<ValidateRequest> requests) {
        List<ProcessParameterQueryService.ValidationResult> results = requests.stream()
                .map(req -> queryService.validateValue(req.getParamId(), req.getValue()))
                .toList();
        return ResponseEntity.ok(results);
    }
    
    @PostMapping("/copy")
    public ResponseEntity<List<ProcessParameterDTO>> copyToRoute(@RequestBody CopyRequest request) {
        List<ProcessParameterDTO> params = commandService.copyToRoute(
                request.getSourceRouteId(),
                request.getTargetRouteId(),
                request.getTargetRouteCode(),
                request.getStepOffset()
        ).stream()
                .map(ProcessParameterDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.status(HttpStatus.CREATED).body(params);
    }
    
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
    
    public static class ValidateRequest {
        private Long paramId;
        private BigDecimal value;
        
        public Long getParamId() { return paramId; }
        public void setParamId(Long paramId) { this.paramId = paramId; }
        public BigDecimal getValue() { return value; }
        public void setValue(BigDecimal value) { this.value = value; }
    }
    
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