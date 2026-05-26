package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class ProductionTask {
    private Long id;
    private String taskNo;
    private Long workOrderId;
    private String workstationId;
    private String processCode;
    private TaskStatus status;
    private Integer quantity;
    private Integer completedQuantity;
    private Integer qualifiedQuantity;
    private Integer defectiveQuantity;
    private String operatorId;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum TaskStatus {
        PENDING, IN_PROGRESS, COMPLETED, QUALITY_CHECK, REWORK, SCRAP
    }

    public void create(String taskNo, Long workOrderId, String workstationId,
                      String processCode, Integer quantity, String operatorId) {
        this.taskNo = taskNo;
        this.workOrderId = workOrderId;
        this.workstationId = workstationId;
        this.processCode = processCode;
        this.quantity = quantity;
        this.operatorId = operatorId;
        this.completedQuantity = 0;
        this.qualifiedQuantity = 0;
        this.defectiveQuantity = 0;
        this.status = TaskStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void start() {
        if (status != TaskStatus.PENDING) {
            throw new IllegalStateException("Can only start PENDING tasks");
        }
        this.status = TaskStatus.IN_PROGRESS;
        this.startTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void complete(Integer qualified, Integer defective) {
        if (status != TaskStatus.IN_PROGRESS) {
            throw new IllegalStateException("Task is not in progress");
        }
        this.qualifiedQuantity = qualified;
        this.defectiveQuantity = defective;
        this.completedQuantity = qualified + defective;
        this.status = TaskStatus.QUALITY_CHECK;
        this.endTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void passQualityCheck() {
        if (status != TaskStatus.QUALITY_CHECK) {
            throw new IllegalStateException("Task is not in quality check");
        }
        this.status = TaskStatus.COMPLETED;
        this.updatedAt = LocalDateTime.now();
    }

    public void failQualityCheck() {
        if (status != TaskStatus.QUALITY_CHECK) {
            throw new IllegalStateException("Task is not in quality check");
        }
        this.status = TaskStatus.REWORK;
        this.updatedAt = LocalDateTime.now();
    }

    public void scrap() {
        this.status = TaskStatus.SCRAP;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTaskNo() { return taskNo; }
    public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
    public Long getWorkOrderId() { return workOrderId; }
    public void setWorkOrderId(Long workOrderId) { this.workOrderId = workOrderId; }
    public String getWorkstationId() { return workstationId; }
    public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
    public String getProcessCode() { return processCode; }
    public void setProcessCode(String processCode) { this.processCode = processCode; }
    public TaskStatus getStatus() { return status; }
    public void setStatus(TaskStatus status) { this.status = status; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public Integer getCompletedQuantity() { return completedQuantity; }
    public void setCompletedQuantity(Integer completedQuantity) { this.completedQuantity = completedQuantity; }
    public Integer getQualifiedQuantity() { return qualifiedQuantity; }
    public void setQualifiedQuantity(Integer qualifiedQuantity) { this.qualifiedQuantity = qualifiedQuantity; }
    public Integer getDefectiveQuantity() { return defectiveQuantity; }
    public void setDefectiveQuantity(Integer defectiveQuantity) { this.defectiveQuantity = defectiveQuantity; }
    public String getOperatorId() { return operatorId; }
    public void setOperatorId(String operatorId) { this.operatorId = operatorId; }
    public LocalDateTime getStartTime() { return startTime; }
    public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
    public LocalDateTime getEndTime() { return endTime; }
    public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}