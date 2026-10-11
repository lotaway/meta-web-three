package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.application.command.MesCommandService;
import com.metawebthree.mes.application.query.MesQueryService;
import com.metawebthree.mes.common.MesPermissions;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/mes")
public class MesController {

    private final MesCommandService commandService;
    private final MesQueryService queryService;

    public MesController(MesCommandService commandService, MesQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // Work Order endpoints
    @PostMapping("/work-order")
    @RequirePermission(MesPermissions.WORK_ORDER_CREATE)
    public ResponseEntity<Map<String, Object>> createWorkOrder(@RequestBody Map<String, Object> request) {
        String workOrderNo = (String) request.get("workOrderNo");
        String productCode = (String) request.get("productCode");
        String productName = (String) request.get("productName");
        Integer quantity = (Integer) request.get("quantity");
        String workshopId = (String) request.get("workshopId");
        String processRouteId = (String) request.get("processRouteId");
        
        Long id = commandService.createWorkOrder(
            workOrderNo, productCode, productName, quantity, workshopId, processRouteId);
        
        return ResponseEntity.ok(Map.of("workOrderId", id));
    }

    @PostMapping("/work-order/{id}/release")
    @RequirePermission(MesPermissions.WORK_ORDER_RELEASE)
    public ResponseEntity<Void> releaseWorkOrder(@PathVariable Long id) {
        commandService.releaseWorkOrder(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/work-order/{id}/start")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<Void> startWorkOrder(@PathVariable Long id) {
        commandService.startWorkOrder(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/work-order/{id}/complete")
    @RequirePermission(MesPermissions.WORK_ORDER_UPDATE)
    public ResponseEntity<Void> completeWorkOrder(@PathVariable Long id) {
        commandService.completeWorkOrder(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/work-order/{id}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<?> getWorkOrder(@PathVariable Long id) {
        return queryService.getWorkOrderById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/work-order/workshop/{workshopId}")
    @RequirePermission(MesPermissions.WORK_ORDER_READ)
    public ResponseEntity<?> getWorkshopWorkOrders(@PathVariable String workshopId) {
        return ResponseEntity.ok(queryService.getWorkshopWorkOrders(workshopId));
    }

    // Task endpoints
    @PostMapping("/task")
    @RequirePermission(MesPermissions.TASK_CREATE)
    public ResponseEntity<Map<String, Object>> createTask(@RequestBody Map<String, Object> request) {
        String taskNo = (String) request.get("taskNo");
        Long workOrderId = ((Number) request.get("workOrderId")).longValue();
        String workstationId = (String) request.get("workstationId");
        String processCode = (String) request.get("processCode");
        Integer quantity = (Integer) request.get("quantity");
        String operatorId = (String) request.get("operatorId");
        
        commandService.createTask(taskNo, workOrderId, workstationId, processCode, quantity, operatorId);
        
        return ResponseEntity.ok(Map.of("taskNo", taskNo));
    }

    @PostMapping("/task/{id}/start")
    @RequirePermission(MesPermissions.TASK_START)
    public ResponseEntity<Void> startTask(@PathVariable Long id) {
        commandService.startTask(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/task/{id}/complete")
    @RequirePermission(MesPermissions.TASK_COMPLETE)
    public ResponseEntity<Void> completeTask(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        Integer qualified = (Integer) request.get("qualified");
        Integer defective = (Integer) request.get("defective");
        commandService.completeTask(id, qualified, defective);
        return ResponseEntity.ok().build();
    }

    // Equipment endpoints
    @GetMapping("/equipment/{id}")
    @RequirePermission(MesPermissions.EQUIPMENT_READ)
    public ResponseEntity<?> getEquipment(@PathVariable Long id) {
        return queryService.getEquipmentById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/equipment/workshop/{workshopId}")
    @RequirePermission(MesPermissions.EQUIPMENT_READ)
    public ResponseEntity<?> getWorkshopEquipment(@PathVariable String workshopId) {
        return ResponseEntity.ok(queryService.getWorkshopEquipment(workshopId));
    }

    @PostMapping("/equipment/{id}/breakdown")
    @RequirePermission(MesPermissions.EQUIPMENT_BREAKDOWN)
    public ResponseEntity<Void> reportBreakdown(@PathVariable Long id) {
        commandService.reportEquipmentBreakdown(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/equipment/{id}/repair")
    @RequirePermission(MesPermissions.EQUIPMENT_REPAIR)
    public ResponseEntity<Void> repairEquipment(@PathVariable Long id) {
        commandService.repairEquipment(id);
        return ResponseEntity.ok().build();
    }
}