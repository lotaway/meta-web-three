package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.application.command.ProductionTaskCommandService;
import com.metawebthree.mes.application.query.ProductionTaskQueryService;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.interfaces.dto.EntityExtensionFieldValueDTO;
import com.metawebthree.mes.interfaces.dto.ProductionTaskDTO;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/production-tasks")
public class ProductionTaskController {
    
    private final ProductionTaskCommandService commandService;
    private final ProductionTaskQueryService queryService;
    
    public ProductionTaskController(
            ProductionTaskCommandService commandService,
            ProductionTaskQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.TASK_CREATE)
    public ResponseEntity<ProductionTaskDTO> create(@RequestBody CreateRequest request) {
        ProductionTask task = commandService.prepareCreateTask(
                request.getTaskNo(),
                request.getWorkOrderId(),
                request.getWorkstationId(),
                request.getProcessCode(),
                request.getQuantity(),
                request.getOperatorId()
        );
        commandService.saveTask(task);
        return ResponseEntity.status(HttpStatus.CREATED).body(ProductionTaskDTO.fromEntity(task));
    }
    
    @PostMapping("/{id}/start")
    @RequirePermission(MesPermissions.TASK_START)
    public ResponseEntity<ProductionTaskDTO> start(@PathVariable Long id) {
        ProductionTask task = commandService.prepareStartTask(id);
        commandService.saveStartTask(task);
        return ResponseEntity.ok(ProductionTaskDTO.fromEntity(task));
    }
    
    @PostMapping("/{id}/complete")
    @RequirePermission(MesPermissions.TASK_COMPLETE)
    public ResponseEntity<ProductionTaskDTO> complete(
            @PathVariable Long id,
            @RequestBody CompleteRequest request) {
        ProductionTask task = commandService.prepareCompleteTask(id, request.getQualified(), request.getDefective());
        commandService.saveCompleteTask(task, request.getQualified(), request.getDefective());
        return ResponseEntity.ok(ProductionTaskDTO.fromEntity(task));
    }
    
    @PostMapping("/{id}/quality/pass")
    public ResponseEntity<ProductionTaskDTO> passQualityCheck(@PathVariable Long id) {
        ProductionTask task = commandService.preparePassQualityCheck(id);
        commandService.savePassQualityCheck(task);
        return ResponseEntity.ok(ProductionTaskDTO.fromEntity(task));
    }
    
    @PostMapping("/{id}/quality/fail")
    public ResponseEntity<ProductionTaskDTO> failQualityCheck(@PathVariable Long id) {
        ProductionTask task = commandService.prepareFailQualityCheck(id);
        commandService.saveFailQualityCheck(task);
        return ResponseEntity.ok(ProductionTaskDTO.fromEntity(task));
    }
    
    @PostMapping("/{id}/scrap")
    public ResponseEntity<ProductionTaskDTO> scrap(@PathVariable Long id) {
        ProductionTask task = commandService.prepareScrapTask(id);
        commandService.saveScrapTask(task);
        return ResponseEntity.ok(ProductionTaskDTO.fromEntity(task));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.TASK_UPDATE)
    public ResponseEntity<ProductionTaskDTO> update(
            @PathVariable Long id,
            @RequestBody UpdateRequest request) {
        ProductionTask task = commandService.prepareUpdateTask(
                id,
                request.getWorkstationId(),
                request.getProcessCode(),
                request.getQuantity(),
                request.getOperatorId()
        );
        commandService.saveUpdateTask(task);
        return ResponseEntity.ok(ProductionTaskDTO.fromEntity(task));
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.TASK_UPDATE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        commandService.deleteTask(id);
        return ResponseEntity.noContent().build();
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.TASK_READ)
    public ResponseEntity<ProductionTaskDTO> getById(@PathVariable Long id) {
        return queryService.findById(id)
                .map(task -> ResponseEntity.ok(ProductionTaskDTO.fromEntity(task)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/no/{taskNo}")
    @RequirePermission(MesPermissions.TASK_READ)
    public ResponseEntity<ProductionTaskDTO> getByTaskNo(@PathVariable String taskNo) {
        return queryService.findByTaskNo(taskNo)
                .map(task -> ResponseEntity.ok(ProductionTaskDTO.fromEntity(task)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/work-order/{workOrderId}")
    @RequirePermission(MesPermissions.TASK_READ)
    public ResponseEntity<List<ProductionTaskDTO>> getByWorkOrderId(@PathVariable Long workOrderId) {
        List<ProductionTaskDTO> tasks = queryService.findByWorkOrderId(workOrderId).stream()
                .map(ProductionTaskDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(tasks);
    }
    
    @GetMapping("/status/{status}")
    @RequirePermission(MesPermissions.TASK_READ)
    public ResponseEntity<List<ProductionTaskDTO>> getByStatus(@PathVariable String status) {
        ProductionTask.TaskStatus taskStatus = ProductionTask.TaskStatus.valueOf(status);
        List<ProductionTaskDTO> tasks = queryService.findByStatus(taskStatus).stream()
                .map(ProductionTaskDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(tasks);
    }
    
    @GetMapping("/workstation/{workstationId}")
    @RequirePermission(MesPermissions.TASK_READ)
    public ResponseEntity<List<ProductionTaskDTO>> getByWorkstationId(@PathVariable String workstationId) {
        List<ProductionTaskDTO> tasks = queryService.findByWorkstationId(workstationId).stream()
                .map(ProductionTaskDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(tasks);
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.TASK_READ)
    public ResponseEntity<List<ProductionTaskDTO>> getAll() {
        List<ProductionTaskDTO> tasks = queryService.findAll().stream()
                .map(ProductionTaskDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(tasks);
    }
    
    @GetMapping("/{id}/extension-values")
    public ResponseEntity<List<EntityExtensionFieldValueDTO>> getExtensionFieldValues(@PathVariable Long id) {
        List<EntityExtensionFieldValueDTO> values = queryService.getExtensionFieldValues(id).stream()
                .map(EntityExtensionFieldValueDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(values);
    }
    
    @PostMapping("/{id}/extension-values")
    public ResponseEntity<Void> setExtensionFieldValues(
            @PathVariable Long id,
            @RequestBody Map<String, String> fieldValues) {
        commandService.setExtensionFieldValues(id, fieldValues);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/work-order/{workOrderId}/count")
    public ResponseEntity<Map<String, Long>> countByWorkOrderId(@PathVariable Long workOrderId) {
        long count = queryService.countByWorkOrderId(workOrderId);
        Map<String, Long> result = new HashMap<>();
        result.put("count", count);
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/status/{status}/count")
    public ResponseEntity<Map<String, Long>> countByStatus(@PathVariable String status) {
        ProductionTask.TaskStatus taskStatus = ProductionTask.TaskStatus.valueOf(status);
        long count = queryService.countByStatus(taskStatus);
        Map<String, Long> result = new HashMap<>();
        result.put("count", count);
        return ResponseEntity.ok(result);
    }
    
    public static class CreateRequest {
        private String taskNo;
        private Long workOrderId;
        private Long workstationId;
        private String processCode;
        private Integer quantity;
        private String operatorId;
        
        public String getTaskNo() { return taskNo; }
        public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
        public Long getWorkOrderId() { return workOrderId; }
        public void setWorkOrderId(Long workOrderId) { this.workOrderId = workOrderId; }
        public Long getWorkstationId() { return workstationId; }
        public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
        public String getProcessCode() { return processCode; }
        public void setProcessCode(String processCode) { this.processCode = processCode; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public String getOperatorId() { return operatorId; }
        public void setOperatorId(String operatorId) { this.operatorId = operatorId; }
    }
    
    public static class UpdateRequest {
        private Long workstationId;
        private String processCode;
        private Integer quantity;
        private String operatorId;
        
        public Long getWorkstationId() { return workstationId; }
        public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
        public String getProcessCode() { return processCode; }
        public void setProcessCode(String processCode) { this.processCode = processCode; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public String getOperatorId() { return operatorId; }
        public void setOperatorId(String operatorId) { this.operatorId = operatorId; }
    }
    
    public static class CompleteRequest {
        private Integer qualified;
        private Integer defective;
        
        public Integer getQualified() { return qualified; }
        public void setQualified(Integer qualified) { this.qualified = qualified; }
        public Integer getDefective() { return defective; }
        public void setDefective(Integer defective) { this.defective = defective; }
    }
}
