package com.metawebthree.mes.interfaces.controller.scheduling;

import com.metawebthree.mes.application.command.SchedulingCommandService;
import com.metawebthree.mes.application.command.SchedulingCommandService.OperationRequest;
import com.metawebthree.mes.application.query.SchedulingQueryService;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResult;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/mes/scheduling")
public class SchedulingController {

    private final SchedulingCommandService commandService;
    private final SchedulingQueryService queryService;

    public SchedulingController(SchedulingCommandService commandService,
                                SchedulingQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping("/orders")
    public ResponseEntity<ScheduleOrder> createOrder(@RequestBody CreateOrderRequest request) {
        ScheduleOrder order = commandService.createScheduleOrder(
            request.getScheduleNo(), request.getOrderNo(), request.getProductCode(),
            request.getProductName(), request.getQuantity(), request.getDueDate(),
            request.getPriority(), request.getWorkshopId(), request.getRouteCode());
        return ResponseEntity.ok(order);
    }

    @PostMapping("/orders/{id}/operations")
    public ResponseEntity<ScheduleOrder> addOperations(@PathVariable Long id,
                                                        @RequestBody List<OperationRequest> operations) {
        ScheduleOrder order = commandService.addOperations(id, operations);
        return ResponseEntity.ok(order);
    }

    @PostMapping("/forward")
    public ResponseEntity<ScheduleResult> scheduleForward(@RequestParam String workshopId) {
        ScheduleResult result = commandService.scheduleForward(workshopId);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/backward")
    public ResponseEntity<ScheduleResult> scheduleBackward(@RequestParam String workshopId) {
        ScheduleResult result = commandService.scheduleBackward(workshopId);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/orders/{id}/reschedule")
    public ResponseEntity<ScheduleResult> reschedule(@PathVariable Long id) {
        ScheduleResult result = commandService.reschedule(id);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/orders/{id}/start")
    public ResponseEntity<ScheduleOrder> startOrder(@PathVariable Long id) {
        ScheduleOrder order = commandService.startOrder(id);
        return ResponseEntity.ok(order);
    }

    @PostMapping("/orders/{id}/complete")
    public ResponseEntity<ScheduleOrder> completeOrder(@PathVariable Long id) {
        ScheduleOrder order = commandService.completeOrder(id);
        return ResponseEntity.ok(order);
    }

    @PostMapping("/orders/{id}/cancel")
    public ResponseEntity<ScheduleOrder> cancelOrder(@PathVariable Long id) {
        ScheduleOrder order = commandService.cancelOrder(id);
        return ResponseEntity.ok(order);
    }

    @PostMapping("/orders/{id}/delay")
    public ResponseEntity<ScheduleOrder> markDelayed(@PathVariable Long id) {
        ScheduleOrder order = commandService.markDelayed(id);
        return ResponseEntity.ok(order);
    }

    @PutMapping("/orders/{id}/progress")
    public ResponseEntity<ScheduleOrder> updateProgress(@PathVariable Long id,
                                                         @RequestBody UpdateProgressRequest request) {
        ScheduleOrder order = commandService.updateProgress(id, request.getCompletedQuantity());
        return ResponseEntity.ok(order);
    }

    @DeleteMapping("/orders/{id}")
    public ResponseEntity<Void> deleteOrder(@PathVariable Long id) {
        commandService.deleteOrder(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/orders/{id}")
    public ResponseEntity<ScheduleOrder> getOrder(@PathVariable Long id) {
        return queryService.findOrderById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/orders")
    public ResponseEntity<List<ScheduleOrder>> listOrders(
            @RequestParam(required = false) String workshopId,
            @RequestParam(required = false) String status) {
        if (workshopId != null) {
            return ResponseEntity.ok(queryService.findOrdersByWorkshop(workshopId));
        } else if (status != null) {
            return ResponseEntity.ok(queryService.findOrdersByStatus(status));
        }
        return ResponseEntity.ok(queryService.findAllOrders());
    }

    @GetMapping("/orders/overdue")
    public ResponseEntity<List<ScheduleOrder>> getOverdueOrders() {
        return ResponseEntity.ok(queryService.findOverdueOrders());
    }

    @PostMapping("/resources")
    public ResponseEntity<ScheduleResource> createResource(@RequestBody CreateResourceRequest request) {
        ScheduleResource resource = commandService.createResource(
            request.getResourceCode(), request.getResourceName(),
            request.getResourceType(), request.getWorkshopId());
        return ResponseEntity.ok(resource);
    }

    @PutMapping("/resources/{id}")
    public ResponseEntity<ScheduleResource> updateResource(@PathVariable Long id,
                                                            @RequestBody UpdateResourceRequest request) {
        ScheduleResource resource = commandService.updateResource(
            id, request.getResourceName(), request.getCapacityPerShift(),
            request.getStatus(), request.getDescription());
        return ResponseEntity.ok(resource);
    }

    @DeleteMapping("/resources/{id}")
    public ResponseEntity<Void> deleteResource(@PathVariable Long id) {
        commandService.deleteResource(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/resources/{id}")
    public ResponseEntity<ScheduleResource> getResource(@PathVariable Long id) {
        return queryService.findResourceById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/resources")
    public ResponseEntity<List<ScheduleResource>> listResources(
            @RequestParam(required = false) String workshopId,
            @RequestParam(required = false) String resourceType) {
        if (workshopId != null) {
            return ResponseEntity.ok(queryService.findResourcesByWorkshop(workshopId));
        } else if (resourceType != null) {
            return ResponseEntity.ok(queryService.findResourcesByType(resourceType));
        }
        return ResponseEntity.ok(queryService.findAllResources());
    }

    public static class CreateOrderRequest {
        private String scheduleNo;
        private String orderNo;
        private String productCode;
        private String productName;
        private BigDecimal quantity;
        private LocalDateTime dueDate;
        private String priority;
        private String workshopId;
        private String routeCode;

        public String getScheduleNo() { return scheduleNo; }
        public void setScheduleNo(String scheduleNo) { this.scheduleNo = scheduleNo; }
        public String getOrderNo() { return orderNo; }
        public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
        public String getProductCode() { return productCode; }
        public void setProductCode(String productCode) { this.productCode = productCode; }
        public String getProductName() { return productName; }
        public void setProductName(String productName) { this.productName = productName; }
        public BigDecimal getQuantity() { return quantity; }
        public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
        public LocalDateTime getDueDate() { return dueDate; }
        public void setDueDate(LocalDateTime dueDate) { this.dueDate = dueDate; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getRouteCode() { return routeCode; }
        public void setRouteCode(String routeCode) { this.routeCode = routeCode; }
    }

    public static class UpdateProgressRequest {
        private BigDecimal completedQuantity;
        public BigDecimal getCompletedQuantity() { return completedQuantity; }
        public void setCompletedQuantity(BigDecimal completedQuantity) { this.completedQuantity = completedQuantity; }
    }

    public static class CreateResourceRequest {
        private String resourceCode;
        private String resourceName;
        private String resourceType;
        private String workshopId;

        public String getResourceCode() { return resourceCode; }
        public void setResourceCode(String resourceCode) { this.resourceCode = resourceCode; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    }

    public static class UpdateResourceRequest {
        private String resourceName;
        private Double capacityPerShift;
        private String status;
        private String description;

        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public Double getCapacityPerShift() { return capacityPerShift; }
        public void setCapacityPerShift(Double capacityPerShift) { this.capacityPerShift = capacityPerShift; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }
}
