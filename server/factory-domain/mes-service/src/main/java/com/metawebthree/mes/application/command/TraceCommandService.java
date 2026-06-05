package com.metawebthree.mes.application.command;

import com.metawebthree.mes.domain.entity.TraceDataScope;
import com.metawebthree.mes.domain.entity.TraceModel;
import com.metawebthree.mes.domain.entity.TraceRecord;
import com.metawebthree.mes.domain.repository.trace.TraceDataScopeRepository;
import com.metawebthree.mes.domain.repository.trace.TraceModelRepository;
import com.metawebthree.mes.domain.repository.trace.TraceRecordRepository;
import com.metawebthree.mes.domain.service.trace.TraceDomainService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TraceCommandService {

    private final TraceRecordRepository traceRecordRepository;
    private final TraceModelRepository traceModelRepository;
    private final TraceDataScopeRepository traceDataScopeRepository;
    private final TraceDomainService traceDomainService;

    public TraceCommandService(TraceRecordRepository traceRecordRepository,
                               TraceModelRepository traceModelRepository,
                               TraceDataScopeRepository traceDataScopeRepository,
                               TraceDomainService traceDomainService) {
        this.traceRecordRepository = traceRecordRepository;
        this.traceModelRepository = traceModelRepository;
        this.traceDataScopeRepository = traceDataScopeRepository;
        this.traceDomainService = traceDomainService;
    }

    // ========== Trace Record ==========
    public TraceRecord createTrace(String traceCode, String traceType, String productCode,
                                    String productName, String batchNo, String sn,
                                    String source, String workOrderNo) {
        return traceDomainService.createTrace(traceCode, traceType, productCode, productName,
            batchNo, sn, source, workOrderNo);
    }

    public TraceRecord linkMaterial(Long traceId, String materialCode, String batchNo, Integer quantity) {
        traceDomainService.linkMaterial(traceId, materialCode, batchNo, quantity);
        return traceRecordRepository.findById(traceId).orElseThrow();
    }

    public TraceRecord linkEquipment(Long traceId, String equipmentCode, String equipmentName) {
        traceDomainService.linkEquipment(traceId, equipmentCode, equipmentName);
        return traceRecordRepository.findById(traceId).orElseThrow();
    }

    public TraceRecord linkOperator(Long traceId, String operatorCode, String operatorName) {
        traceDomainService.linkOperator(traceId, operatorCode, operatorName);
        return traceRecordRepository.findById(traceId).orElseThrow();
    }

    public TraceRecord linkQcResult(Long traceId, String qcRecordCode) {
        traceDomainService.linkQcResult(traceId, qcRecordCode);
        return traceRecordRepository.findById(traceId).orElseThrow();
    }

    public TraceRecord.TraceChain buildTraceChain(String traceCode) {
        return traceDomainService.buildTraceChain(traceCode);
    }

    public void deleteTrace(Long id) {
        traceRecordRepository.deleteById(id);
    }

    // ========== Trace Model ==========
    public TraceModel createModel(String modelCode, String modelName, String productType) {
        TraceModel model = new TraceModel();
        model.create(modelCode, modelName, productType);
        return traceModelRepository.save(model);
    }

    public TraceModel updateModel(Long id, String modelName, String productType,
                                   Boolean enableBatch, Boolean enableSn,
                                   Boolean enableMaterial, Boolean enableProcess,
                                   Boolean enableQuality, Boolean enableEquipment) {
        TraceModel model = traceModelRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Trace model not found: " + id));
        model.setModelName(modelName);
        model.setProductType(productType);
        if (enableBatch != null) model.enableBatchTrace(enableBatch);
        if (enableSn != null) model.enableSnTrace(enableSn);
        if (enableMaterial != null) model.enableMaterialTrace(enableMaterial);
        if (enableProcess != null) model.enableProcessTrace(enableProcess);
        if (enableQuality != null) model.enableQualityTrace(enableQuality);
        if (enableEquipment != null) model.enableEquipmentTrace(enableEquipment);
        traceModelRepository.update(model);
        return model;
    }

    public void deleteModel(Long id) {
        traceModelRepository.deleteById(id);
    }

    // ========== Trace Data Scope ==========
    public TraceDataScope createDataScope(String scopeCode, String scopeName,
                                           TraceDataScope.DataScopeType scopeType) {
        TraceDataScope scope = new TraceDataScope();
        scope.create(scopeCode, scopeName, scopeType);
        return traceDataScopeRepository.save(scope);
    }

    public TraceDataScope updateDataScope(Long id, String scopeName,
                                           Boolean isDefault) {
        TraceDataScope scope = traceDataScopeRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Trace data scope not found: " + id));
        scope.setScopeName(scopeName);
        if (isDefault != null) scope.setIsDefault(isDefault);
        traceDataScopeRepository.update(scope);
        return scope;
    }

    public void deleteDataScope(Long id) {
        traceDataScopeRepository.deleteById(id);
    }
}
