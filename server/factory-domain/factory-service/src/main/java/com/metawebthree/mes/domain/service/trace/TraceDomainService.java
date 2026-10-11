package com.metawebthree.mes.domain.service.trace;

import com.metawebthree.mes.domain.entity.TraceRecord;

import java.util.List;

public interface TraceDomainService {
    TraceRecord createTrace(String traceCode, String traceType, String productCode, String productName,
                             String batchNo, String sn, String source, String workOrderNo);
    void linkMaterial(Long traceId, String materialCode, String batchNo, Integer quantity);
    void linkEquipment(Long traceId, String equipmentCode, String equipmentName);
    void linkOperator(Long traceId, String operatorCode, String operatorName);
    void linkQcResult(Long traceId, String qcRecordCode);
    void linkChildBatch(Long traceId, String childBatchNo, Integer quantity);
    List<TraceRecord> forwardTrace(String traceCode);
    List<TraceRecord> backwardTrace(String traceCode);
    TraceRecord.TraceChain buildTraceChain(String traceCode);
}
