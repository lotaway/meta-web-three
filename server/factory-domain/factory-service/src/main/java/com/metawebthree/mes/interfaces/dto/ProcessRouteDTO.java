package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import com.metawebthree.mes.domain.entity.ProcessRoute;

public class ProcessRouteDTO {
    
    private Long id;
    private String routeCode;
    private String routeName;
    private String productCode;
    private Integer version;
    private String status;
    private List<ProcessStepDTO> steps;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private String validationMessage;
    private Boolean validationResult;
    
    public static ProcessRouteDTO fromEntity(ProcessRoute entity) {
        if (entity == null) return null;
        
        ProcessRouteDTO dto = new ProcessRouteDTO();
        dto.setId(entity.getId());
        dto.setRouteCode(entity.getRouteCode());
        dto.setRouteName(entity.getRouteName());
        dto.setProductCode(entity.getProductCode());
        dto.setVersion(entity.getVersion());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        if (entity.getSteps() != null) {
            dto.setSteps(entity.getSteps().stream()
                .map(ProcessStepDTO::fromEntity)
                .collect(Collectors.toList()));
        }
        
        return dto;
    }
    
    public ProcessRoute toEntity() {
        ProcessRoute route = new ProcessRoute();
        if (this.id != null) {
            route.setId(this.id);
        }
        route.setRouteCode(this.routeCode);
        route.setRouteName(this.routeName);
        route.setProductCode(this.productCode);
        
        if (this.steps != null) {
            route.setSteps(this.steps.stream()
                .map(ProcessStepDTO::toEntity)
                .collect(Collectors.toList()));
        }
        
        return route;
    }
    
    // ========== Request DTOs ==========
    
    public static class CreateRequest {
        private String routeCode;
        private String routeName;
        private String productCode;
        private List<ProcessStepDTO> steps;
        
        public String getRouteCode() { return routeCode; }
        public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
        public String getRouteName() { return routeName; }
        public void setRouteName(String routeName) { this.routeName = routeName; }
        public String getProductCode() { return productCode; }
        public void setProductCode(String productCode) { this.productCode = productCode; }
        public List<ProcessStepDTO> getSteps() { return steps; }
        public void setSteps(List<ProcessStepDTO> steps) { this.steps = steps; }
    }
    
    public static class UpdateRequest {
        private String routeName;
        private String productCode;
        private List<ProcessStepDTO> steps;
        
        public String getRouteName() { return routeName; }
        public void setRouteName(String routeName) { this.routeName = routeName; }
        public String getProductCode() { return productCode; }
        public void setProductCode(String productCode) { this.productCode = productCode; }
        public List<ProcessStepDTO> getSteps() { return steps; }
        public void setSteps(List<ProcessStepDTO> steps) { this.steps = steps; }
    }
    
    // ========== ProcessStep DTO ==========
    
    public static class ProcessStepDTO {
        private Integer stepNo;
        private String processCode;
        private String processName;
        private Long workstationId;
        private Integer standardTime;
        private String qualityCheckpoint;
        private Integer predecessorStepNo;
        private Integer successorStepNo;
        
        public static ProcessStepDTO fromEntity(ProcessRoute.ProcessStep entity) {
            if (entity == null) return null;
            
            ProcessStepDTO dto = new ProcessStepDTO();
            dto.setStepNo(entity.getStepNo());
            dto.setProcessCode(entity.getProcessCode());
            dto.setProcessName(entity.getProcessName());
            dto.setWorkstationId(entity.getWorkstationId());
            dto.setStandardTime(entity.getStandardTime());
            dto.setQualityCheckpoint(entity.getQualityCheckpoint());
            dto.setPredecessorStepNo(entity.getPredecessorStepNo());
            dto.setSuccessorStepNo(entity.getSuccessorStepNo());
            return dto;
        }
        
        public ProcessRoute.ProcessStep toEntity() {
            ProcessRoute.ProcessStep step = new ProcessRoute.ProcessStep();
            step.setStepNo(this.stepNo);
            step.setProcessCode(this.processCode);
            step.setProcessName(this.processName);
            step.setWorkstationId(this.workstationId);
            step.setStandardTime(this.standardTime);
            step.setQualityCheckpoint(this.qualityCheckpoint);
            step.setPredecessorStepNo(this.predecessorStepNo);
            step.setSuccessorStepNo(this.successorStepNo);
            return step;
        }
        
        public Integer getStepNo() { return stepNo; }
        public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
        public String getProcessCode() { return processCode; }
        public void setProcessCode(String processCode) { this.processCode = processCode; }
        public String getProcessName() { return processName; }
        public void setProcessName(String processName) { this.processName = processName; }
        public Long getWorkstationId() { return workstationId; }
        public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
        public Integer getStandardTime() { return standardTime; }
        public void setStandardTime(Integer standardTime) { this.standardTime = standardTime; }
        public String getQualityCheckpoint() { return qualityCheckpoint; }
        public void setQualityCheckpoint(String qualityCheckpoint) { this.qualityCheckpoint = qualityCheckpoint; }
        public Integer getPredecessorStepNo() { return predecessorStepNo; }
        public void setPredecessorStepNo(Integer predecessorStepNo) { this.predecessorStepNo = predecessorStepNo; }
        public Integer getSuccessorStepNo() { return successorStepNo; }
        public void setSuccessorStepNo(Integer successorStepNo) { this.successorStepNo = successorStepNo; }
    }
    
    // ========== Getters and Setters ==========
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRouteCode() { return routeCode; }
    public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
    public String getRouteName() { return routeName; }
    public void setRouteName(String routeName) { this.routeName = routeName; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public Integer getVersion() { return version; }
    public void setVersion(Integer version) { this.version = version; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public List<ProcessStepDTO> getSteps() { return steps; }
    public void setSteps(List<ProcessStepDTO> steps) { this.steps = steps; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public String getValidationMessage() { return validationMessage; }
    public void setValidationMessage(String validationMessage) { this.validationMessage = validationMessage; }
    public Boolean getValidationResult() { return validationResult; }
    public void setValidationResult(Boolean validationResult) { this.validationResult = validationResult; }
}
