package com.metawebthree.mes.domain.repository.trace;

import com.metawebthree.mes.domain.entity.TraceRecord;
import java.util.List;
import java.util.Optional;

public interface TraceRecordRepository {
    Optional<TraceRecord> findById(Long id);
    Optional<TraceRecord> findByTraceCode(String traceCode);
    List<TraceRecord> findByBatchNo(String batchNo);
    List<TraceRecord> findByProductCode(String productCode);
    List<TraceRecord> findBySn(String sn);
    List<TraceRecord> findByWorkOrderNo(String workOrderNo);
    List<TraceRecord> findBySource(String source);
    List<TraceRecord> findAll();
    TraceRecord save(TraceRecord record);
    void update(TraceRecord record);
    void deleteById(Long id);
}
