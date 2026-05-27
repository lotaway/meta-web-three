package com.metawebthree.mes.domain.entity;

import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.service.StatusMachineService;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public class WorkOrder {
    private Long id;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private Integer completedQuantity;
    private WorkOrderStatus status;
    private String statusCode; // 可配置状态机的状态码
    private String typeCode; // 工单类型: NORMAL, REWORK, REPAIR, SAMPLE
    private Priority priority;
    private String workshopId;
    private String processRouteId;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // 静态服务引用（通过setter注入）
    private static StatusMachineService statusMachineService;
    
    // 默认状态机编码
    private static final String DEFAULT_STATUS_MACHINE_CODE = "WORK_ORDER_DEFAULT";

    public enum WorkOrderStatus {
        DRAFT, RELEASED, IN_PROGRESS, PAUSED, COMPLETED, CANCELLED
    }
    
    // 兼容：旧的状态码到枚举的映射
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
    
    // 注入状态机服务（用于单元测试模拟）
    public static void setStatusMachineService(StatusMachineService service) {
        statusMachineService = service;
    }

    public void create(String workOrderNo, String productCode, String productName, 
                      Integer quantity, String workshopId, String processRouteId) {
        this.workOrderNo = workOrderNo;
        this.productCode = productCode;
        this.productName = productName;
        this.quantity = quantity;
        this.workshopId = workshopId;
        this.processRouteId = processRouteId;
        this.completedQuantity = 0;
        this.status = WorkOrderStatus.DRAFT;
        this.statusCode = "DRAFT";
        this.typeCode = "NORMAL"; // 默认工单类型
        this.priority = Priority.NORMAL;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 使用工单类型创建工单
     */
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
        this.statusCode = "IN_PROGRESS";
        this.actualStartTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void pause() {
        validateTransition("PAUSE");
        this.status = WorkOrderStatus.PAUSED;
        this.statusCode = "PAUSED";
        this.updatedAt = LocalDateTime.now();
    }

    public void resume() {
        validateTransition("RESUME");
        this.status = WorkOrderStatus.IN_PROGRESS;
        this.statusCode = "IN_PROGRESS";
        this.updatedAt = LocalDateTime.now();
    }

    public void complete() {
        validateTransition("COMPLETE");
        if (completedQuantity < quantity) {
            throw new IllegalStateException("Cannot complete: not all quantities finished");
        }
        this.status = WorkOrderStatus.COMPLETED;
        this.statusCode = "COMPLETED";
        this.actualEndTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void cancel() {
        validateTransition("CANCEL");
        this.status = WorkOrderStatus.CANCELLED;
        this.statusCode = "CANCELLED";
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 带原因取消工单
     */
    public void cancelWithReason(String reason) {
        validateTransition("CANCEL");
        this.status = WorkOrderStatus.CANCELLED;
        this.statusCode = "CANCELLED";
        this.updatedAt = LocalDateTime.now();
    }

    public void updateProgress(Integer quantity) {
        if (status != WorkOrderStatus.IN_PROGRESS) {
            throw new IllegalStateException("Work order is not in progress");
        }
        this.completedQuantity += quantity;
        if (this.completedQuantity >= this.quantity) {
            this.completedQuantity = this.quantity;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public Double getCompletionRate() {
        if (quantity == 0) return 0.0;
        return (double) completedQuantity / quantity * 100;
    }
    
    /**
     * 验证状态转换是否允许
     */
    private void validateTransition(String action) {
        // 如果没有配置状态机服务，使用默认的枚举验证
        if (statusMachineService == null) {
            validateLegacyTransition(action);
            return;
        }
        
        // 尝试获取可配置的状态机
        Optional<StatusMachine> machineOpt = statusMachineService.getStatusMachine("WORK_ORDER");
        if (machineOpt.isEmpty()) {
            // 回退到传统验证
            validateLegacyTransition(action);
            return;
        }
        
        StatusMachine machine = machineOpt.get();
        String currentStatus = statusCode != null ? statusCode : status.name();
        
        // 检查转换是否有效
        boolean isValid = machine.getTransitions().stream()
                .anyMatch(t -> t.getFromStatus().equals(currentStatus) 
                           && t.getTransitionAction().equals(action));
        
        if (!isValid) {
            throw new IllegalStateException(
                String.format("Cannot perform action '%s' from status '%s'", action, currentStatus));
        }
    }
    
    /**
     * 传统枚举验证（向后兼容）
     */
    private void validateLegacyTransition(String action) {
        switch (action) {
            case "RELEASE":
                if (status != WorkOrderStatus.DRAFT) {
                    throw new IllegalStateException("Can only release DRAFT work orders");
                }
                break;
            case "START":
                if (status != WorkOrderStatus.RELEASED) {
                    throw new IllegalStateException("Can only start RELEASED work orders");
                }
                break;
            case "PAUSE":
                if (status != WorkOrderStatus.IN_PROGRESS) {
                    throw new IllegalStateException("Can only pause IN_PROGRESS work orders");
                }
                break;
            case "RESUME":
                if (status != WorkOrderStatus.PAUSED) {
                    throw new IllegalStateException("Can only resume PAUSED work orders");
                }
                break;
            case "COMPLETE":
                if (status != WorkOrderStatus.IN_PROGRESS) {
                    throw new IllegalStateException("Can only complete IN_PROGRESS work orders");
                }
                break;
            case "CANCEL":
                if (status == WorkOrderStatus.COMPLETED) {
                    throw new IllegalStateException("Cannot cancel completed work orders");
                }
                break;
            default:
                throw new IllegalArgumentException("Unknown action: " + action);
        }
    }
    
    /**
     * 获取当前状态码（优先使用可配置的状态码）
     */
    public String getStatusCode() {
        return statusCode != null ? statusCode : (status != null ? status.name() : null);
    }
    
    /**
     * 获取工单类型
     */
    public String getTypeCode() {
        return typeCode;
    }
    
    public void setTypeCode(String typeCode) {
        this.typeCode = typeCode;
    }
    
    public void setStatusCode(String statusCode) {
        this.statusCode = statusCode;
        // 同步更新枚举状态
        if (statusCode != null && STATUS_CODE_MAP.containsKey(statusCode)) {
            this.status = STATUS_CODE_MAP.get(statusCode);
        }
    }
    
    // Getters and Setters
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
    public WorkOrderStatus getStatus() { return status; }
    public void setStatus(WorkOrderStatus status) { 
        this.status = status; 
        if (status != null) {
            this.statusCode = status.name();
        }
    }
    public Priority getPriority() { return priority; }
    public void setPriority(Priority priority) { this.priority = priority; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
    public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
    public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
    public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}