package com.metawebthree.mes.domain.config;

import lombok.Data;

@Data
public class StatusConfig {
    private Long id;
    private Long machineId;
    private String statusCode;
    private String statusName;
    private String statusCategory;
    private Boolean isInitial;
    private Boolean isFinal;
    private String color;
    private String icon;
    private Integer sortOrder;

    // Explicit getters and setters (Lombok annotation processor not working)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getMachineId() { return machineId; }
    public void setMachineId(Long machineId) { this.machineId = machineId; }
    public String getStatusCode() { return statusCode; }
    public void setStatusCode(String statusCode) { this.statusCode = statusCode; }
    public String getStatusName() { return statusName; }
    public void setStatusName(String statusName) { this.statusName = statusName; }
    public String getStatusCategory() { return statusCategory; }
    public void setStatusCategory(String statusCategory) { this.statusCategory = statusCategory; }
    public Boolean getIsInitial() { return isInitial; }
    public void setIsInitial(Boolean isInitial) { this.isInitial = isInitial; }
    public Boolean getIsFinal() { return isFinal; }
    public void setIsFinal(Boolean isFinal) { this.isFinal = isFinal; }
    public String getColor() { return color; }
    public void setColor(String color) { this.color = color; }
    public String getIcon() { return icon; }
    public void setIcon(String icon) { this.icon = icon; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
}