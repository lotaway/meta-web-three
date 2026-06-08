package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.MesPermissions;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.application.command.TraceCommandService;
import com.metawebthree.mes.application.query.TraceQueryService;
import com.metawebthree.mes.domain.entity.TraceDataScope;
import com.metawebthree.mes.domain.entity.TraceModel;
import com.metawebthree.mes.domain.entity.TraceRecord;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/mes/trace")
public class TraceController {

    private final TraceCommandService commandService;
    private final TraceQueryService queryService;

    public TraceController(TraceCommandService commandService, TraceQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // ========== Trace Records ==========
    @PostMapping("/records")
    @RequirePermission(MesPermissions.TRACE_CREATE)
    public ResponseEntity<TraceRecord> createRecord(@RequestBody CreateTraceRequest req) {
        return ResponseEntity.ok(commandService.createTrace(
                req.getTraceCode(), req.getTraceType(), req.getProductCode(),
                req.getProductName(), req.getBatchNo(), req.getSn(),
                req.getSource(), req.getWorkOrderNo()));
    }

    @PostMapping("/records/{id}/material")
    @RequirePermission(MesPermissions.TRACE_UPDATE)
    public ResponseEntity<TraceRecord> linkMaterial(@PathVariable Long id, @RequestBody LinkMaterialRequest req) {
        return ResponseEntity
                .ok(commandService.linkMaterial(id, req.getMaterialCode(), req.getBatchNo(), req.getQuantity()));
    }

    @PostMapping("/records/{id}/equipment")
    @RequirePermission(MesPermissions.TRACE_UPDATE)
    public ResponseEntity<TraceRecord> linkEquipment(@PathVariable Long id, @RequestBody LinkEquipmentRequest req) {
        return ResponseEntity.ok(commandService.linkEquipment(id, req.getEquipmentCode(), req.getEquipmentName()));
    }

    @PostMapping("/records/{id}/operator")
    @RequirePermission(MesPermissions.TRACE_UPDATE)
    public ResponseEntity<TraceRecord> linkOperator(@PathVariable Long id, @RequestBody LinkOperatorRequest req) {
        return ResponseEntity.ok(commandService.linkOperator(id, req.getOperatorCode(), req.getOperatorName()));
    }

    @PostMapping("/records/{id}/qc")
    @RequirePermission(MesPermissions.TRACE_UPDATE)
    public ResponseEntity<TraceRecord> linkQc(@PathVariable Long id, @RequestBody LinkQcRequest req) {
        return ResponseEntity.ok(commandService.linkQcResult(id, req.getQcRecordCode()));
    }

    @GetMapping("/records/{id}")
    @RequirePermission(MesPermissions.TRACE_READ)
    public ResponseEntity<TraceRecord> getRecord(@PathVariable Long id) {
        return queryService.findRecordById(id)
                .map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/records/trace-chain")
    @RequirePermission(MesPermissions.TRACE_CHAIN)
    public ResponseEntity<TraceRecord.TraceChain> getTraceChain(@RequestParam String traceCode) {
        return ResponseEntity.ok(commandService.buildTraceChain(traceCode));
    }

    @GetMapping("/records/forward")
    @RequirePermission(MesPermissions.TRACE_FORWARD)
    public ResponseEntity<List<TraceRecord>> forwardTrace(@RequestParam String traceCode) {
        return ResponseEntity.ok(queryService.forwardTrace(traceCode));
    }

    @GetMapping("/records/backward")
    @RequirePermission(MesPermissions.TRACE_BACKWARD)
    public ResponseEntity<List<TraceRecord>> backwardTrace(@RequestParam String traceCode) {
        return ResponseEntity.ok(queryService.backwardTrace(traceCode));
    }

    @GetMapping("/records")
    @RequirePermission(MesPermissions.TRACE_READ)
    public ResponseEntity<List<TraceRecord>> listRecords(
            @RequestParam(required = false) String productCode,
            @RequestParam(required = false) String batchNo,
            @RequestParam(required = false) String workOrderNo,
            @RequestParam(required = false) String sn) {
        if (productCode != null)
            return ResponseEntity.ok(queryService.findRecordsByProduct(productCode));
        if (batchNo != null)
            return ResponseEntity.ok(queryService.findRecordsByBatch(batchNo));
        if (workOrderNo != null)
            return ResponseEntity.ok(queryService.findRecordsByWorkOrder(workOrderNo));
        if (sn != null)
            return ResponseEntity.ok(queryService.findRecordsBySn(sn));
        return ResponseEntity.ok(queryService.findAllRecords());
    }

    @DeleteMapping("/records/{id}")
    @RequirePermission(MesPermissions.TRACE_DELETE)
    public ResponseEntity<Void> deleteRecord(@PathVariable Long id) {
        commandService.deleteTrace(id);
        return ResponseEntity.ok().build();
    }

    // ========== Trace Models ==========
    @PostMapping("/models")
    @RequirePermission(MesPermissions.TRACE_CREATE)
    public ResponseEntity<TraceModel> createModel(@RequestBody CreateModelRequest req) {
        return ResponseEntity
                .ok(commandService.createModel(req.getModelCode(), req.getModelName(), req.getProductType()));
    }

    @PutMapping("/models/{id}")
    @RequirePermission(MesPermissions.TRACE_UPDATE)
    public ResponseEntity<TraceModel> updateModel(@PathVariable Long id, @RequestBody UpdateModelRequest req) {
        return ResponseEntity.ok(commandService.updateModel(id, req.getModelName(), req.getProductType(),
                req.getEnableBatch(), req.getEnableSn(), req.getEnableMaterial(),
                req.getEnableProcess(), req.getEnableQuality(), req.getEnableEquipment()));
    }

    @GetMapping("/models/{id}")
    @RequirePermission(MesPermissions.TRACE_READ)
    public ResponseEntity<TraceModel> getModel(@PathVariable Long id) {
        return queryService.findModelById(id)
                .map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/models")
    @RequirePermission(MesPermissions.TRACE_READ)
    public ResponseEntity<List<TraceModel>> listModels(
            @RequestParam(required = false) String productType) {
        if (productType != null)
            return ResponseEntity.ok(queryService.findModelsByProductType(productType));
        return ResponseEntity.ok(queryService.findAllModels());
    }

    @DeleteMapping("/models/{id}")
    @RequirePermission(MesPermissions.TRACE_DELETE)
    public ResponseEntity<Void> deleteModel(@PathVariable Long id) {
        commandService.deleteModel(id);
        return ResponseEntity.ok().build();
    }

    // ========== Trace Data Scopes ==========
    @PostMapping("/data-scopes")
    @RequirePermission(MesPermissions.TRACE_CREATE)
    public ResponseEntity<TraceDataScope> createDataScope(@RequestBody CreateDataScopeRequest req) {
        return ResponseEntity
                .ok(commandService.createDataScope(req.getScopeCode(), req.getScopeName(), req.getScopeType()));
    }

    @PutMapping("/data-scopes/{id}")
    @RequirePermission(MesPermissions.TRACE_UPDATE)
    public ResponseEntity<TraceDataScope> updateDataScope(@PathVariable Long id,
            @RequestBody UpdateDataScopeRequest req) {
        return ResponseEntity.ok(commandService.updateDataScope(id, req.getScopeName(), req.getIsDefault()));
    }

    @GetMapping("/data-scopes/{id}")
    @RequirePermission(MesPermissions.TRACE_READ)
    public ResponseEntity<TraceDataScope> getDataScope(@PathVariable Long id) {
        return queryService.findScopeById(id)
                .map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/data-scopes")
    @RequirePermission(MesPermissions.TRACE_READ)
    public ResponseEntity<List<TraceDataScope>> listDataScopes(
            @RequestParam(required = false) String scopeType) {
        if (scopeType != null)
            return ResponseEntity.ok(queryService.findScopesByType(TraceDataScope.DataScopeType.valueOf(scopeType)));
        return ResponseEntity.ok(queryService.findAllScopes());
    }

    @DeleteMapping("/data-scopes/{id}")
    @RequirePermission(MesPermissions.TRACE_DELETE)
    public ResponseEntity<Void> deleteDataScope(@PathVariable Long id) {
        commandService.deleteDataScope(id);
        return ResponseEntity.ok().build();
    }

    // ========== Inner DTOs ==========
    public static class CreateTraceRequest {
        private String traceCode;
        private String traceType;
        private String productCode;
        private String productName;
        private String batchNo;
        private String sn;
        private String source;
        private String workOrderNo;

        public String getTraceCode() {
            return traceCode;
        }

        public void setTraceCode(String traceCode) {
            this.traceCode = traceCode;
        }

        public String getTraceType() {
            return traceType;
        }

        public void setTraceType(String traceType) {
            this.traceType = traceType;
        }

        public String getProductCode() {
            return productCode;
        }

        public void setProductCode(String productCode) {
            this.productCode = productCode;
        }

        public String getProductName() {
            return productName;
        }

        public void setProductName(String productName) {
            this.productName = productName;
        }

        public String getBatchNo() {
            return batchNo;
        }

        public void setBatchNo(String batchNo) {
            this.batchNo = batchNo;
        }

        public String getSn() {
            return sn;
        }

        public void setSn(String sn) {
            this.sn = sn;
        }

        public String getSource() {
            return source;
        }

        public void setSource(String source) {
            this.source = source;
        }

        public String getWorkOrderNo() {
            return workOrderNo;
        }

        public void setWorkOrderNo(String workOrderNo) {
            this.workOrderNo = workOrderNo;
        }
    }

    public static class LinkMaterialRequest {
        private String materialCode;
        private String batchNo;
        private Integer quantity;

        public String getMaterialCode() {
            return materialCode;
        }

        public void setMaterialCode(String materialCode) {
            this.materialCode = materialCode;
        }

        public String getBatchNo() {
            return batchNo;
        }

        public void setBatchNo(String batchNo) {
            this.batchNo = batchNo;
        }

        public Integer getQuantity() {
            return quantity;
        }

        public void setQuantity(Integer quantity) {
            this.quantity = quantity;
        }
    }

    public static class LinkEquipmentRequest {
        private String equipmentCode;
        private String equipmentName;

        public String getEquipmentCode() {
            return equipmentCode;
        }

        public void setEquipmentCode(String equipmentCode) {
            this.equipmentCode = equipmentCode;
        }

        public String getEquipmentName() {
            return equipmentName;
        }

        public void setEquipmentName(String equipmentName) {
            this.equipmentName = equipmentName;
        }
    }

    public static class LinkOperatorRequest {
        private String operatorCode;
        private String operatorName;

        public String getOperatorCode() {
            return operatorCode;
        }

        public void setOperatorCode(String operatorCode) {
            this.operatorCode = operatorCode;
        }

        public String getOperatorName() {
            return operatorName;
        }

        public void setOperatorName(String operatorName) {
            this.operatorName = operatorName;
        }
    }

    public static class LinkQcRequest {
        private String qcRecordCode;

        public String getQcRecordCode() {
            return qcRecordCode;
        }

        public void setQcRecordCode(String qcRecordCode) {
            this.qcRecordCode = qcRecordCode;
        }
    }

    public static class CreateModelRequest {
        private String modelCode;
        private String modelName;
        private String productType;

        public String getModelCode() {
            return modelCode;
        }

        public void setModelCode(String modelCode) {
            this.modelCode = modelCode;
        }

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getProductType() {
            return productType;
        }

        public void setProductType(String productType) {
            this.productType = productType;
        }
    }

    public static class UpdateModelRequest {
        private String modelName;
        private String productType;
        private Boolean enableBatch;
        private Boolean enableSn;
        private Boolean enableMaterial;
        private Boolean enableProcess;
        private Boolean enableQuality;
        private Boolean enableEquipment;

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getProductType() {
            return productType;
        }

        public void setProductType(String productType) {
            this.productType = productType;
        }

        public Boolean getEnableBatch() {
            return enableBatch;
        }

        public void setEnableBatch(Boolean enableBatch) {
            this.enableBatch = enableBatch;
        }

        public Boolean getEnableSn() {
            return enableSn;
        }

        public void setEnableSn(Boolean enableSn) {
            this.enableSn = enableSn;
        }

        public Boolean getEnableMaterial() {
            return enableMaterial;
        }

        public void setEnableMaterial(Boolean enableMaterial) {
            this.enableMaterial = enableMaterial;
        }

        public Boolean getEnableProcess() {
            return enableProcess;
        }

        public void setEnableProcess(Boolean enableProcess) {
            this.enableProcess = enableProcess;
        }

        public Boolean getEnableQuality() {
            return enableQuality;
        }

        public void setEnableQuality(Boolean enableQuality) {
            this.enableQuality = enableQuality;
        }

        public Boolean getEnableEquipment() {
            return enableEquipment;
        }

        public void setEnableEquipment(Boolean enableEquipment) {
            this.enableEquipment = enableEquipment;
        }
    }

    public static class CreateDataScopeRequest {
        private String scopeCode;
        private String scopeName;
        private TraceDataScope.DataScopeType scopeType;

        public String getScopeCode() {
            return scopeCode;
        }

        public void setScopeCode(String scopeCode) {
            this.scopeCode = scopeCode;
        }

        public String getScopeName() {
            return scopeName;
        }

        public void setScopeName(String scopeName) {
            this.scopeName = scopeName;
        }

        public TraceDataScope.DataScopeType getScopeType() {
            return scopeType;
        }

        public void setScopeType(TraceDataScope.DataScopeType scopeType) {
            this.scopeType = scopeType;
        }
    }

    public static class UpdateDataScopeRequest {
        private String scopeName;
        private Boolean isDefault;

        public String getScopeName() {
            return scopeName;
        }

        public void setScopeName(String scopeName) {
            this.scopeName = scopeName;
        }

        public Boolean getIsDefault() {
            return isDefault;
        }

        public void setIsDefault(Boolean isDefault) {
            this.isDefault = isDefault;
        }
    }
}
