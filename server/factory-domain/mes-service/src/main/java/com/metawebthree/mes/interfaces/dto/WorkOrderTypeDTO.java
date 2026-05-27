package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;

public class WorkOrderTypeDTO {
    
    private Long id;
    private String typeCode;
    private String typeName;
    private String description;
    private String statusMachineCode;
    private String processRouteTemplate;
    private Boolean isDefault;
    private Integer sortOrder;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static WorkOrderTypeDTO fromEntity(
            com.metawebthree.mes.domain.config.WorkOrderType entity) {
        if (entity == null) return null;
        
        WorkOrderTypeDTO dto = new WorkOrderTypeDTO();
        dto.setId(entity.getId());
        dto.setTypeCode(entity.getTypeCode());
        dto.setTypeName(entity.getTypeName());
        dto.setDescription(entity.getDescription());
        dto.setStatusMachineCode(entity.getStatusMachineCode());
        dto.setProcessRouteTemplate(entity.getProcessRouteTemplate());
        dto.setIsDefault(entity.getIsDefault());
        dto.setSortOrder(entity.getSortOrder());
        dto.setStatus(entity.getStatus());
        dto.setCreatedAt(null); // WorkOrderType entity doesn't have createdAt
        dto.setUpdatedAt(null);
        return dto;
    }
    
    public com.metawebthree.mes.domain.config.WorkOrderType toEntity() {
        com.metawebthree.mes.domain.config.WorkOrderType entity = new com.metawebthree.mes.domain.config.WorkOrderType();
        entity.setId(this.id);
        entity.setTypeCode(this.typeCode);
        entity.setTypeName(this.typeName);
        entity.setDescription(this.description);
        entity.setStatusMachineCode(this.statusMachineCode);
        entity.setProcessRouteTemplate(this.processRouteTemplate);
        entity.setIsDefault(this.isDefault);
        entity.setSortOrder(this.sortOrder);
        entity.setStatus(this.status);
        return entity;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getTypeName() { return typeName; }
    public void setTypeName(String typeName) { this.typeName = typeName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getStatusMachineCode() { return statusMachineCode; }
    public void setStatusMachineCode(String statusMachineCode) { this.statusMachineCode = statusMachineCode; }
    public String getProcessRouteTemplate() { return processRouteTemplate; }
    public void setProcessRouteTemplate(String processRouteTemplate) { this.processRouteTemplate = processRouteTemplate; }
    public Boolean getIsDefault() { return isDefault; }
    public void setIsDefault(Boolean isDefault) { this.isDefault = isDefault; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}