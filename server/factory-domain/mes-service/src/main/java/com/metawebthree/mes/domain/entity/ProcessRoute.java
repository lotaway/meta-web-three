package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.List;

public class ProcessRoute {
    private Long id;
    private String routeCode;
    private String routeName;
    private String productCode;
    private Integer version;
    private RouteStatus status;
    private List<ProcessStep> steps;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum RouteStatus {
        DRAFT, ACTIVE, ARCHIVED
    }

    public static class ProcessStep {
        private Integer stepNo;
        private String processCode;
        private String processName;
        private String workstationId;
        private Integer standardTime;
        private String qualityCheckpoint;

        public Integer getStepNo() { return stepNo; }
        public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
        public String getProcessCode() { return processCode; }
        public void setProcessCode(String processCode) { this.processCode = processCode; }
        public String getProcessName() { return processName; }
        public void setProcessName(String processName) { this.processName = processName; }
        public String getWorkstationId() { return workstationId; }
        public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
        public Integer getStandardTime() { return standardTime; }
        public void setStandardTime(Integer standardTime) { this.standardTime = standardTime; }
        public String getQualityCheckpoint() { return qualityCheckpoint; }
        public void setQualityCheckpoint(String qualityCheckpoint) { this.qualityCheckpoint = qualityCheckpoint; }
    }

    public void create(String routeCode, String routeName, String productCode) {
        this.routeCode = routeCode;
        this.routeName = routeName;
        this.productCode = productCode;
        this.version = 1;
        this.status = RouteStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.status = RouteStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void archive() {
        this.status = RouteStatus.ARCHIVED;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateVersion() {
        this.version++;
        this.updatedAt = LocalDateTime.now();
    }

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
    public RouteStatus getStatus() { return status; }
    public void setStatus(RouteStatus status) { this.status = status; }
    public List<ProcessStep> getSteps() { return steps; }
    public void setSteps(List<ProcessStep> steps) { this.steps = steps; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}