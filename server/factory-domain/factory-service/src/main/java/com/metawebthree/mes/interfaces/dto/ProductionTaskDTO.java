package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.ProductionTask;

import java.time.LocalDateTime;

public class ProductionTaskDTO {
    private Long id;
    private String taskNo;
    private Long workOrderId;
    private Long workstationId;
    private String processCode;
    private String status;
    private Integer quantity;
    private Integer completedQuantity;
    private Integer qualifiedQuantity;
    private Integer defectiveQuantity;
    private String operatorId;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static ProductionTaskDTO fromEntity(ProductionTask task) {
        ProductionTaskDTO dto = new ProductionTaskDTO();
        dto.setId(task.getId());
        dto.setTaskNo(task.getTaskNo());
        dto.setWorkOrderId(task.getWorkOrderId());
        dto.setWorkstationId(task.getWorkstationId());
        dto.setProcessCode(task.getProcessCode());
        dto.setStatus(task.getStatus() != null ? task.getStatus().name() : null);
        dto.setQuantity(task.getQuantity());
        dto.setCompletedQuantity(task.getCompletedQuantity());
        dto.setQualifiedQuantity(task.getQualifiedQuantity());
        dto.setDefectiveQuantity(task.getDefectiveQuantity());
        dto.setOperatorId(task.getOperatorId());
        dto.setStartTime(task.getStartTime());
        dto.setEndTime(task.getEndTime());
        dto.setCreatedAt(task.getCreatedAt());
        dto.setUpdatedAt(task.getUpdatedAt());
        return dto;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTaskNo() { return taskNo; }
    public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
    public Long getWorkOrderId() { return workOrderId; }
    public void setWorkOrderId(Long workOrderId) { this.workOrderId = workOrderId; }
    public Long getWorkstationId() { return workstationId; }
    public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
    public String getProcessCode() { return processCode; }
    public void setProcessCode(String processCode) { this.processCode = processCode; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
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
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}