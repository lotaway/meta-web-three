package com.metawebthree.mes.interfaces.dto;

import jakarta.validation.constraints.*;

public class UpdateWorkOrderTypeRequest {
    
    @Size(max = 100, message = "typeName长度不能超过100")
    private String typeName;
    
    @Size(max = 500, message = "description长度不能超过500")
    private String description;
    
    @Size(max = 50, message = "statusMachineCode长度不能超过50")
    private String statusMachineCode;
    
    @Size(max = 50, message = "processRouteTemplate长度不能超过50")
    private String processRouteTemplate;
    
    private Boolean isDefault;
    
    @Min(value = 0, message = "sortOrder不能为负数")
    @Max(value = 999, message = "sortOrder不能超过999")
    private Integer sortOrder;
    
    @Size(max = 20, message = "status长度不能超过20")
    private String status;
    
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
}