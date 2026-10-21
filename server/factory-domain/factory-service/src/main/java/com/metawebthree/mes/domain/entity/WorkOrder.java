package com.metawebthree.mes.domain.entity;

import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.service.StatusMachineService;

import java.math.BigDecimal;
import java.math.RoundingMode;
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
    private String remark;
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

    /**
     * 工单使用的状态机编码。
     */
    public String getStatusMachineCode() {
        return DEFAULT_STATUS_MACHINE_CODE;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void resume() {
        validateTransition("RESUME");
        this.status = WorkOrderStatus.IN_PROGRESS;
        this.statusCode = "IN_PROGRESS";
        this.updatedAt = LocalDateTime.now();
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
        this.statusCode = "COMPLETED";
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * 取消工单。
     */
    public void cancel() {
        this.status = WorkOrderStatus.CANCELLED;
        this.statusCode = WorkOrderStatus.CANCELLED.name();
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * 取消工单并记录原因。
     *
     * @param reason 取消原因
     */
    public void cancelWithReason(String reason) {
        cancel();
        this.remark = reason;
    }

    /**
     * 更新完成数量。
     *
     * @param quantity 本次新增的完成数量（非负）
     */
    public void updateProgress(Integer quantity) {
        if (quantity == null || quantity < 0) {
            throw new IllegalArgumentException("Completed quantity must not be negative, got: " + quantity);
        }
        int current = this.completedQuantity != null ? this.completedQuantity : 0;
        int updated = current + quantity;
        if (this.quantity != null && updated > this.quantity) {
            throw new IllegalStateException(
                    "Completed quantity (" + updated + ") exceeds work order quantity (" + this.quantity + ")");
        }
        this.completedQuantity = updated;
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * 完成率（百分比，保留两位小数）。
     */
    public Double getCompletionRate() {
        if (quantity == null || quantity <= 0) {
            return 0.0d;
        }
        int completed = completedQuantity != null ? completedQuantity : 0;
        return BigDecimal.valueOf(completed)
                .multiply(BigDecimal.valueOf(100))
                .divide(BigDecimal.valueOf(quantity), 2, RoundingMode.HALF_UP)
                .doubleValue();
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

    public String getRemark() {
        return remark;
    }

    public void setRemark(String remark) {
        this.remark = remark;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}