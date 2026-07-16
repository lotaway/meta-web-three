package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.application.command.WorkOrderCommandService;
import com.metawebthree.mes.application.query.WorkOrderQueryService;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.interfaces.dto.WorkOrderDTO;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/work-orders")
public class WorkOrderController {
    
    private final WorkOrderCommandService commandService;
    private final WorkOrderQueryService queryService;
    
    public WorkOrderController(
            WorkOrderCommandService commandService,
            WorkOrderQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.WORK_ORDER_CREATE)
    public ResponseEntity<WorkOrderDTO> create(@RequestBody CreateRequest request) {
        WorkOrder workOrder;
        if (request.getTypeCode() != null && !request.getTypeCode().isEmpty()) {
            workOrder = commandService.prepareCreateWorkOrderWithType(
                    request.getWorkOrderNo(),
                    request.getProductCode(),
                    request.getProductName(),
                    request.getQuantity(),
                    request.getWorkshopId(),
                    request.getProcessRouteId(),
                    request.getTypeCode()
            );
        } else {
            workOrder = commandService.prepareCreateWorkOrder(
                    request.getWorkOrderNo(),
                    request.getProductCode(),
                    request.getProductName(),
                    request.getQuantity(),
                    request.getWorkshopId(),
                    request.getProcessRouteId()
            );
        }
        commandService.saveWorkOrder(workOrder);
        return ResponseEntity.status(HttpStatus.CREATED).body(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> update(
            @PathVariable Long id,
            @RequestBody UpdateRequest request) {
        WorkOrder workOrder = commandService.prepareUpdateOrder(
                id,
                request.getProductCode(),
                request.getProductName(),
                request.getQuantity(),
                request.getWorkshopId(),
                request.getProcessRouteId(),
                request.getPriority(),
                request.getPlannedStartTime(),
                request.getPlannedEndTime()
        );
        commandService.saveUpdateOrder(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/release")
    @RequirePermission(MesPermissions.WORK_ORDER_RELEASE)
    public ResponseEntity<WorkOrderDTO> release(@PathVariable Long id) {
        WorkOrder workOrder = commandService.prepareRelease(id);
        commandService.saveRelease(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/start")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> start(@PathVariable Long id) {
        WorkOrder workOrder = commandService.prepareStart(id);
        commandService.saveStart(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/pause")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> pause(@PathVariable Long id) {
        WorkOrder workOrder = commandService.preparePause(id);
        commandService.savePause(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/resume")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> resume(@PathVariable Long id) {
        WorkOrder workOrder = commandService.prepareResume(id);
        commandService.saveResume(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/complete")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> complete(@PathVariable Long id) {
        WorkOrder workOrder = commandService.prepareComplete(id);
        commandService.saveComplete(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/cancel")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> cancel(
            @PathVariable Long id,
            @RequestBody(required = false) CancelRequest request) {
        WorkOrder workOrder;
        if (request != null && request.getReason() != null) {
            workOrder = commandService.prepareCancelWithReason(id, request.getReason());
        } else {
            workOrder = commandService.prepareCancel(id);
        }
        commandService.saveCancel(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/progress")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<WorkOrderDTO> updateProgress(
            @PathVariable Long id,
            @RequestBody ProgressRequest request) {
        WorkOrder workOrder = commandService.prepareUpdateProgress(id, request.getQuantity());
        commandService.saveUpdateProgress(workOrder);
        return ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder));
    }
    
    @PostMapping("/{id}/split")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<List<WorkOrderDTO>> split(
            @PathVariable Long id,
            @RequestBody SplitRequest request) {
        List<WorkOrder> childOrders = commandService.prepareSplit(
                id, request.getSplitType(), request.getSplitCount()
        );
        commandService.saveSplitOrders(childOrders);
        List<WorkOrderDTO> childDTOs = childOrders.stream()
                .map(WorkOrderDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(childDTOs);
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.WORK_ORDER_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        commandService.deleteWorkOrder(id);
        return ResponseEntity.noContent().build();
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<WorkOrderDTO> getById(@PathVariable Long id) {
        return queryService.findById(id)
                .map(workOrder -> ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/no/{workOrderNo}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<WorkOrderDTO> getByWorkOrderNo(@PathVariable String workOrderNo) {
        return queryService.findByWorkOrderNo(workOrderNo)
                .map(workOrder -> ResponseEntity.ok(WorkOrderDTO.fromEntity(workOrder)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/status/{status}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<List<WorkOrderDTO>> getByStatus(@PathVariable String status) {
        WorkOrder.WorkOrderStatus workOrderStatus = WorkOrder.WorkOrderStatus.valueOf(status);
        List<WorkOrderDTO> workOrders = queryService.findByStatus(workOrderStatus).stream()
                .map(WorkOrderDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(workOrders);
    }
    
    @GetMapping("/workshop/{workshopId}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<List<WorkOrderDTO>> getByWorkshopId(@PathVariable String workshopId) {
        List<WorkOrderDTO> workOrders = queryService.findByWorkshopId(workshopId).stream()
                .map(WorkOrderDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(workOrders);
    }
    
    @GetMapping("/product/{productCode}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<List<WorkOrderDTO>> getByProductCode(@PathVariable String productCode) {
        List<WorkOrderDTO> workOrders = queryService.findByProductCode(productCode).stream()
                .map(WorkOrderDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(workOrders);
    }
    
    @GetMapping("/parent/{parentWorkOrderId}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<List<WorkOrderDTO>> getChildOrders(@PathVariable Long parentWorkOrderId) {
        List<WorkOrderDTO> workOrders = queryService.findByParentWorkOrderId(parentWorkOrderId).stream()
                .map(WorkOrderDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(workOrders);
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<List<WorkOrderDTO>> getAll() {
        List<WorkOrderDTO> workOrders = queryService.findAll().stream()
                .map(WorkOrderDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(workOrders);
    }
    
    public static class CreateRequest {
        private String workOrderNo;
        private String productCode;
        private String productName;
        private Integer quantity;
        private String workshopId;
        private String processRouteId;
        private String typeCode;
        
        public String getWorkOrderNo() { return workOrderNo; }
        public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
        public String getProductCode() { return productCode; }
        public void setProductCode(String productCode) { this.productCode = productCode; }
        public String getProductName() { return productName; }
        public void setProductName(String productName) { this.productName = productName; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getProcessRouteId() { return processRouteId; }
        public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
        public String getTypeCode() { return typeCode; }
        public void setTypeCode(String typeCode) { this.typeCode = typeCode; }
    }
    
    public static class UpdateRequest {
        private String productCode;
        private String productName;
        private Integer quantity;
        private String workshopId;
        private String processRouteId;
        private String priority;
        private LocalDateTime plannedStartTime;
        private LocalDateTime plannedEndTime;
        
        public String getProductCode() { return productCode; }
        public void setProductCode(String productCode) { this.productCode = productCode; }
        public String getProductName() { return productName; }
        public void setProductName(String productName) { this.productName = productName; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getProcessRouteId() { return processRouteId; }
        public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
        public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
        public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
        public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    }
    
    public static class CancelRequest {
        private String reason;
        
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
    }
    
    public static class ProgressRequest {
        private Integer quantity;
        
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
    }
    
    public static class SplitRequest {
        private String splitType;
        private Integer splitCount;
        
        public String getSplitType() { return splitType; }
        public void setSplitType(String splitType) { this.splitType = splitType; }
        public Integer getSplitCount() { return splitCount; }
        public void setSplitCount(Integer splitCount) { this.splitCount = splitCount; }
    }
}
