package com.metawebthree.mes.application.query;

import com.metawebthree.mes.domain.entity.TraceDataScope;
import com.metawebthree.mes.domain.entity.TraceModel;
import com.metawebthree.mes.domain.entity.TraceRecord;
import com.metawebthree.mes.domain.repository.trace.TraceDataScopeRepository;
import com.metawebthree.mes.domain.repository.trace.TraceModelRepository;
import com.metawebthree.mes.domain.repository.trace.TraceRecordRepository;
import com.metawebthree.mes.domain.service.trace.TraceDomainService;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class TraceQueryService {

    private final TraceRecordRepository traceRecordRepository;
    private final TraceModelRepository traceModelRepository;
    private final TraceDataScopeRepository traceDataScopeRepository;
    private final TraceDomainService traceDomainService;

    public TraceQueryService(TraceRecordRepository traceRecordRepository,
                             TraceModelRepository traceModelRepository,
                             TraceDataScopeRepository traceDataScopeRepository,
                             TraceDomainService traceDomainService) {
        this.traceRecordRepository = traceRecordRepository;
        this.traceModelRepository = traceModelRepository;
        this.traceDataScopeRepository = traceDataScopeRepository;
        this.traceDomainService = traceDomainService;
    }

    // Trace Records
    public Optional<TraceRecord> findRecordById(Long id) { return traceRecordRepository.findById(id); }
    public Optional<TraceRecord> findRecordByCode(String code) { return traceRecordRepository.findByTraceCode(code); }
    public List<TraceRecord> findRecordsByBatch(String batchNo) { return traceRecordRepository.findByBatchNo(batchNo); }
    public List<TraceRecord> findRecordsByProduct(String productCode) { return traceRecordRepository.findByProductCode(productCode); }
    public List<TraceRecord> findRecordsBySn(String sn) { return traceRecordRepository.findBySn(sn); }
    public List<TraceRecord> findRecordsByWorkOrder(String workOrderNo) { return traceRecordRepository.findByWorkOrderNo(workOrderNo); }
    public List<TraceRecord> findAllRecords() { return traceRecordRepository.findAll(); }

    // Trace Chain
    public List<TraceRecord> forwardTrace(String traceCode) { return traceDomainService.forwardTrace(traceCode); }
    public List<TraceRecord> backwardTrace(String traceCode) { return traceDomainService.backwardTrace(traceCode); }

    // Trace Models
    public Optional<TraceModel> findModelById(Long id) { return traceModelRepository.findById(id); }
    public Optional<TraceModel> findModelByCode(String code) { return traceModelRepository.findByModelCode(code); }
    public List<TraceModel> findModelsByProductType(String productType) { return traceModelRepository.findByProductType(productType); }
    public List<TraceModel> findAllModels() { return traceModelRepository.findAll(); }

    // Trace Data Scopes
    public Optional<TraceDataScope> findScopeById(Long id) { return traceDataScopeRepository.findById(id); }
    public Optional<TraceDataScope> findScopeByCode(String code) { return traceDataScopeRepository.findByScopeCode(code); }
    public List<TraceDataScope> findScopesByType(TraceDataScope.DataScopeType type) { return traceDataScopeRepository.findByScopeType(type); }
    public List<TraceDataScope> findAllScopes() { return traceDataScopeRepository.findAll(); }
}
