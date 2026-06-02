package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.WorkOrderService;
import com.metawebthree.cs.domain.model.WorkOrder;
import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import com.metawebthree.cs.domain.model.enums.WorkOrderStatus;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/cs/work-order")
@Tag(name = "Work Order Controller", description = "工单管理接口")
public class WorkOrderController {
    private final WorkOrderService workOrderService;

    public WorkOrderController(WorkOrderService workOrderService) {
        this.workOrderService = workOrderService;
    }

    @Operation(summary = "创建工单")
    @PostMapping("/create")
    public ApiResponse<WorkOrder> create(@RequestBody WorkOrderRequest request) {
        WorkOrder workOrder;
        if (request.getCategory() != null) {
            workOrder = workOrderService.createWorkOrder(
                    request.getCustomerId(), request.getTitle(), 
                    request.getDescription(), request.getCategory());
        } else {
            workOrder = workOrderService.createWorkOrder(
                    request.getCustomerId(), request.getTitle(), request.getDescription());
        }
        return ApiResponse.success(workOrder);
    }

    @Operation(summary = "根据ID查询工单")
    @GetMapping("/get")
    public ApiResponse<WorkOrder> get(@RequestParam Long id) {
        return ApiResponse.success(workOrderService.getWorkOrder(id).orElse(null));
    }

    @Operation(summary = "查询客户工单列表")
    @GetMapping("/customer")
    public ApiResponse<List<WorkOrder>> getByCustomer(@RequestParam Long customerId) {
        return ApiResponse.success(workOrderService.getCustomerWorkOrders(customerId));
    }

    @Operation(summary = "查询客服工单列表")
    @GetMapping("/agent")
    public ApiResponse<List<WorkOrder>> getByAgent(@RequestParam Long agentId) {
        return ApiResponse.success(workOrderService.getAgentWorkOrders(agentId));
    }

    @Operation(summary = "分配工单给客服")
    @PostMapping("/assign")
    public ApiResponse<WorkOrder> assign(@RequestParam Long workOrderId, @RequestParam Long agentId) {
        return ApiResponse.success(workOrderService.assignAgent(workOrderId, agentId));
    }

    @Operation(summary = "解决工单")
    @PostMapping("/resolve")
    public ApiResponse<WorkOrder> resolve(@RequestParam Long workOrderId, @RequestParam String resolution) {
        return ApiResponse.success(workOrderService.resolveWorkOrder(workOrderId, resolution));
    }

    @Operation(summary = "升级工单")
    @PostMapping("/escalate")
    public ApiResponse<WorkOrder> escalate(@RequestParam Long workOrderId) {
        return ApiResponse.success(workOrderService.escalateWorkOrder(workOrderId));
    }

    @Operation(summary = "重新分类工单")
    @PostMapping("/reclassify")
    public ApiResponse<WorkOrder> reclassify(@RequestParam Long workOrderId, 
                                              @RequestParam String title, 
                                              @RequestParam String description) {
        return ApiResponse.success(workOrderService.reclassify(workOrderId, title, description));
    }

    @Operation(summary = "查询待处理工单")
    @GetMapping("/pending")
    public ApiResponse<List<WorkOrder>> pending() {
        return ApiResponse.success(workOrderService.getPendingWorkOrders());
    }

    @Operation(summary = "按状态查询工单")
    @GetMapping("/byStatus")
    public ApiResponse<List<WorkOrder>> byStatus(@RequestParam WorkOrderStatus status) {
        return ApiResponse.success(workOrderService.getWorkOrdersByStatus(status));
    }

    @Operation(summary = "按分类查询工单")
    @GetMapping("/byCategory")
    public ApiResponse<List<WorkOrder>> byCategory(@RequestParam WorkOrderCategory category) {
        return ApiResponse.success(workOrderService.getWorkOrdersByCategory(category));
    }

    @Operation(summary = "按状态统计工单数量")
    @GetMapping("/countByStatus")
    public ApiResponse<Long> countByStatus(@RequestParam WorkOrderStatus status) {
        return ApiResponse.success(workOrderService.countByStatus(status));
    }

    @Operation(summary = "按分类统计工单数量")
    @GetMapping("/countByCategory")
    public ApiResponse<Long> countByCategory(@RequestParam WorkOrderCategory category) {
        return ApiResponse.success(workOrderService.countByCategory(category));
    }

    public static class WorkOrderRequest {
        private Long customerId;
        private String title;
        private String description;
        private WorkOrderCategory category;

        public Long getCustomerId() { return customerId; }
        public void setCustomerId(Long customerId) { this.customerId = customerId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public WorkOrderCategory getCategory() { return category; }
        public void setCategory(WorkOrderCategory category) { this.category = category; }
    }
}