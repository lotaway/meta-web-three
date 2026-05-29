package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.WorkOrder;
import java.time.LocalDateTime;

public class WorkOrderDTO {
    private Long id;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private Integer completedQuantity;
    private String status;
    private String statusCode;
    private String typeCode;
    private String priority;
    private String workshopId;
    private String processRouteId;
    private Long parentWorkOrderId;
    private Long splitRuleId;
    private Integer splitSequence;
    private String splitType;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Double completionRate;
    
    public static WorkOrderDTO fromEntity(WorkOrder workOrder) {
        WorkOrderDTO dto = new WorkOrderDTO();
        dto.setId(workOrder.getId());
        dto.setWorkOrderNo(workOrder.getWorkOrderNo());
        dto.setProductCode(workOrder.getProductCode());
        dto.setProductName(workOrder.getProductName());
        dto.setQuantity(workOrder.getQuantity());
        dto.setCompletedQuantity(workOrder.getCompletedQuantity());
        dto.setStatus(workOrder.getStatus() != null ? workOrder.getStatus().name() : null);
        dto.setStatusCode(workOrder.getStatusCode());
        dto.setTypeCode(workOrder.getTypeCode());
        dto.setPriority(workOrder.getPriority() != null ? workOrder.getPriority().name() : null);
        dto.setWorkshopId(workOrder.getWorkshopId());
        dto.setProcessRouteId(workOrder.getProcessRouteId());
        dto.setParentWorkOrderId(workOrder.getParentWorkOrderId());
        dto.setSplitRuleId(workOrder.getSplitRuleId());
        dto.setSplitSequence(workOrder.getSplitSequence());
        dto.setSplitType(workOrder.getSplitType());
        dto.setPlannedStartTime(workOrder.getPlannedStartTime());
        dto.setPlannedEndTime(workOrder.getPlannedEndTime());
        dto.setActualStartTime(workOrder.getActualStartTime());
        dto.setActualEndTime(workOrder.getActualEndTime());
        dto.setCreatedAt(workOrder.getCreatedAt());
        dto.setUpdatedAt(workOrder.getUpdatedAt());
        dto.setCompletionRate(workOrder.getCompletionRate());
        return dto;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getWorkOrderNo() { return workOrderNo; }
    public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public Integer getCompletedQuantity() { return completedQuantity; }
    public void setCompletedQuantity(Integer completedQuantity) { this.completedQuantity = completedQuantity; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getStatusCode() { return statusCode; }
    public void setStatusCode(String statusCode) { this.statusCode = statusCode; }
    public String getTypeCode() { return typeCode; }
    public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    public String getPriority() { return priority; }
    public void setPriority(String priority) { this.priority = priority; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public Long getParentWorkOrderId() { return parentWorkOrderId; }
    public void setParentWorkOrderId(Long parentWorkOrderId) { this.parentWorkOrderId = parentWorkOrderId; }
    public Long getSplitRuleId() { return splitRuleId; }
    public void setSplitRuleId(Long splitRuleId) { this.splitRuleId = splitRuleId; }
    public Integer getSplitSequence() { return splitSequence; }
    public void setSplitSequence(Integer splitSequence) { this.splitSequence = splitSequence; }
    public String getSplitType() { return splitType; }
    public void setSplitType(String splitType) { this.splitType = splitType; }
    public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
    public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
    public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
    public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public Double getCompletionRate() { return completionRate; }
    public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
}