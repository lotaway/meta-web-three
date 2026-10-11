package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class SopRouteBinding {
    private Long id;
    private Long sopDocumentId;
    private String routeCode;
    private String routeName;
    private Integer stepNo;
    private String processCode;
    private String processName;
    private Long workstationId;
    private String workstationName;
    private Integer sortOrder;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long sopDocumentId, String routeCode, Integer stepNo, Long workstationId) {
        this.sopDocumentId = sopDocumentId;
        this.routeCode = routeCode;
        this.stepNo = stepNo;
        this.workstationId = workstationId;
        this.isActive = true;
        this.sortOrder = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getSopDocumentId() { return sopDocumentId; }
    public void setSopDocumentId(Long sopDocumentId) { this.sopDocumentId = sopDocumentId; }
    public String getRouteCode() { return routeCode; }
    public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
    public String getRouteName() { return routeName; }
    public void setRouteName(String routeName) { this.routeName = routeName; }
    public Integer getStepNo() { return stepNo; }
    public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
    public String getProcessCode() { return processCode; }
    public void setProcessCode(String processCode) { this.processCode = processCode; }
    public String getProcessName() { return processName; }
    public void setProcessName(String processName) { this.processName = processName; }
    public Long getWorkstationId() { return workstationId; }
    public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
    public String getWorkstationName() { return workstationName; }
    public void setWorkstationName(String workstationName) { this.workstationName = workstationName; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public Boolean getIsActive() { return isActive; }
    public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}