package com.metawebthree.mes.domain.entity;

import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.service.StatusMachineService;

import java.time.LocalDateTime;
import java.util.Optional;

public class WorkOrder {
    private Long id;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private Integer completedQuantity;
    private WorkOrderStatus status;
    private String statusCode;
    private String typeCode;
    private Priority priority;
    private String workshopId;
    private String processRouteId;
    private Long codeRuleId;
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
    
    private transient StatusMachineService statusMachineService;
    
    private static final String DEFAULT_STATUS_MACHINE_CODE = "WORK_ORDER_DEFAULT";

    public enum WorkOrderStatus {
        DRAFT, RELEASED, IN_PROGRESS, PAUSED, COMPLETED, CANCELLED, PENDING
    }
    
    public static final java.util.Map<String, WorkOrderStatus> STATUS_CODE_MAP = 
        java.util.Map.of(
            "DRAFT", WorkOrderStatus.DRAFT,
            "RELEASED", WorkOrderStatus.RELEASED,
            "IN_PROGRESS", WorkOrderStatus.IN_PROGRESS,
            "PAUSED", WorkOrderStatus.PAUSED,
            "COMPLETED", WorkOrderStatus.COMPLETED,
            "CANCELLED", WorkOrderStatus.CANCELLED
        );

    public enum Priority {
        LOW, NORMAL, HIGH, URGENT
    }
    
    public enum SplitType {
        BY_QUANTITY,
        BY_BOM,
        BY_PROCESS,
        MANUAL
    }
    
    public void setStatusMachineService(StatusMachineService service) {
        this.statusMachineService = service;
    }

    public void create(String workOrderNo, String productCode, String productName,
                      Integer quantity, String workshopId, String processRouteId) {
        if (quantity == null || quantity <= 0) {
            throw new IllegalArgumentException("WorkOrder quantity must be positive, got: " + quantity);
        }
        if (workshopId == null || workshopId.isBlank()) {
            throw new IllegalArgumentException("WorkOrder workshopId must not be null/blank");
        }
        if (processRouteId == null || processRouteId.isBlank()) {
            throw new IllegalArgumentException("WorkOrder processRouteId must not be null/blank");
        }
        this.workOrderNo = workOrderNo;
        this.productCode = productCode;
        this.productName = productName;
        this.quantity = quantity;
        this.workshopId = workshopId;
        this.processRouteId = processRouteId;
        this.completedQuantity = 0;
        this.status = WorkOrderStatus.DRAFT;
        this.statusCode = "DRAFT";
        this.typeCode = "NORMAL";
        this.priority = Priority.NORMAL;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void createWithType(String workOrderNo, String productCode, String productName,
                               Integer quantity, String workshopId, String processRouteId,
                               String typeCode) {
        create(workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        this.typeCode = typeCode;
    }

    public void release() {
        validateTransition("RELEASE");
        this.status = WorkOrderStatus.RELEASED;
        this.statusCode = "RELEASED";
        this.updatedAt = LocalDateTime.now();
    }

    public void start() {
        validateTransition("START");
        this.status = WorkOrderStatus.IN_PROGRESS;
        this.updatedAt = LocalDateTime.now();
    }

    public void pause() {
        validateTransition("PAUSE");
        this.status = WorkOrderStatus.PAUSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void complete() {
        validateTransition("COMPLETE");
        this.status = WorkOrderStatus.COMPLETED;
        this.updatedAt = LocalDateTime.now();
    }

    private void validateTransition(String transition) {
        if (statusMachineService == null) {
            throw new IllegalStateException("StatusMachineService not configured");
        }
        statusMachineService.validateTransition(this, transition);
    }

    public void setWorkOrderNo(String workOrderNo) {
        this.workOrderNo = workOrderNo;
    }

    public String getWorkOrderNo() {
        return workOrderNo;
    }

    public void setProductCode(String productCode) {
        this.productCode = productCode;
    }

    public String getProductCode() {
        return productCode;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public String getProductName() {
        return productName;
    }

    public Integer getQuantity() {
        return quantity;
    }

    public void setQuantity(Integer quantity) {
        this.quantity = quantity;
    }

    public Integer getCompletedQuantity() {
        return completedQuantity;
    }

    public void setCompletedQuantity(Integer completedQuantity) {
        this.completedQuantity = completedQuantity;
    }

    public WorkOrderStatus getStatus() {
        return status;
    }

    public void setStatus(WorkOrderStatus status) {
        this.status = status;
    }

    public String getStatusCode() {
        return statusCode;
    }

    public void setStatusCode(String statusCode) {
        this.statusCode = statusCode;
    }

    public String getTypeCode() {
        return typeCode;
    }

    public void setTypeCode(String typeCode) {
        this.typeCode = typeCode;
    }

    public Priority getPriority() {
        return priority;
    }

    public void setPriority(Priority priority) {
        this.priority = priority;
    }

    public String getWorkshopId() {
        return workshopId;
    }

    public void setWorkshopId(String workshopId) {
        this.workshopId = workshopId;
    }

    public String getProcessRouteId() {
        return processRouteId;
    }

    public void setProcessRouteId(String processRouteId) {
        this.processRouteId = processRouteId;
    }

    public Long getCodeRuleId() {
        return codeRuleId;
    }

    public void setCodeRuleId(Long codeRuleId) {
        this.codeRuleId = codeRuleId;
    }

    public Long getParentWorkOrderId() {
        return parentWorkOrderId;
    }

    public void setParentWorkOrderId(Long parentWorkOrderId) {
        this.parentWorkOrderId = parentWorkOrderId;
    }

    public Long getSplitRuleId() {
        return splitRuleId;
    }

    public void setSplitRuleId(Long splitRuleId) {
        this.splitRuleId = splitRuleId;
    }

    public Integer getSplitSequence() {
        return splitSequence;
    }

    public void setSplitSequence(Integer splitSequence) {
        this.splitSequence = splitSequence;
    }

    public String getSplitType() {
        return splitType;
    }

    public void setSplitType(String splitType) {
        this.splitType = splitType;
    }

    public LocalDateTime getPlannedStartTime() {
        return plannedStartTime;
    }

    public void setPlannedStartTime(LocalDateTime plannedStartTime) {
        this.plannedStartTime = plannedStartTime;
    }

    public LocalDateTime getPlannedEndTime() {
        return plannedEndTime;
    }

    public void setPlannedEndTime(LocalDateTime plannedEndTime) {
        this.plannedEndTime = plannedEndTime;
    }

    public LocalDateTime getActualStartTime() {
        return actualStartTime;
    }

    public void setActualStartTime(LocalDateTime actualStartTime) {
        this.actualStartTime = actualStartTime;
    }

    public LocalDateTime getActualEndTime() {
        return actualEndTime;
    }

    public void setActualEndTime(LocalDateTime actualEndTime) {
        this.actualEndTime = actualEndTime;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
}