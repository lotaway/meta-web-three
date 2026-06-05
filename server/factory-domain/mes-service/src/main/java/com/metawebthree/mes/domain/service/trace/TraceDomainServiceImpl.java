package com.metawebthree.mes.domain.service.trace;

import com.metawebthree.mes.domain.entity.TraceRecord;
import com.metawebthree.mes.domain.entity.TraceRecord.TraceRelation;
import com.metawebthree.mes.domain.entity.TraceRecord.TraceSource;
import com.metawebthree.mes.domain.entity.TraceRecord.TraceType;
import com.metawebthree.mes.domain.repository.trace.TraceRecordRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

@Slf4j
@Service
public class TraceDomainServiceImpl implements TraceDomainService {

    private final TraceRecordRepository traceRecordRepository;

    public TraceDomainServiceImpl(TraceRecordRepository traceRecordRepository) {
        this.traceRecordRepository = traceRecordRepository;
    }

    @Override
    public TraceRecord createTrace(String traceCode, String traceType, String productCode,
                                    String productName, String batchNo, String sn,
                                    String source, String workOrderNo) {
        TraceRecord record = new TraceRecord();
        record.create(traceCode, TraceType.valueOf(traceType), productCode);
        record.setBatchNo(batchNo);
        record.setSn(sn);
        record.setSource(traceType != null ? TraceSource.valueOf(source) : null);
        return traceRecordRepository.save(record);
    }

    @Override
    public void linkMaterial(Long traceId, String materialCode, String batchNo, Integer quantity) {
        TraceRecord record = traceRecordRepository.findById(traceId)
            .orElseThrow(() -> new IllegalArgumentException("Trace record not found: " + traceId));
        record.linkMaterial(materialCode, batchNo, quantity);
        traceRecordRepository.update(record);
        log.info("Linked material {} to trace {}", materialCode, record.getTraceCode());
    }

    @Override
    public void linkEquipment(Long traceId, String equipmentCode, String equipmentName) {
        TraceRecord record = traceRecordRepository.findById(traceId)
            .orElseThrow(() -> new IllegalArgumentException("Trace record not found: " + traceId));
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(equipmentCode);
        relation.setRelatedType(TraceType.EQUIPMENT);
        relation.setRelationType("BIND");
        record.getRelations().add(relation);
        traceRecordRepository.update(record);
    }

    @Override
    public void linkOperator(Long traceId, String operatorCode, String operatorName) {
        TraceRecord record = traceRecordRepository.findById(traceId)
            .orElseThrow(() -> new IllegalArgumentException("Trace record not found: " + traceId));
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(operatorCode);
        relation.setRelatedType(TraceType.OPERATOR);
        relation.setRelationType("BIND");
        record.getRelations().add(relation);
        traceRecordRepository.update(record);
    }

    @Override
    public void linkQcResult(Long traceId, String qcRecordCode) {
        TraceRecord record = traceRecordRepository.findById(traceId)
            .orElseThrow(() -> new IllegalArgumentException("Trace record not found: " + traceId));
        TraceRelation relation = new TraceRelation();
        relation.setRelatedCode(qcRecordCode);
        relation.setRelatedType(TraceType.QC);
        relation.setRelationType("BIND");
        record.getRelations().add(relation);
        traceRecordRepository.update(record);
    }

    @Override
    public void linkChildBatch(Long traceId, String childBatchNo, Integer quantity) {
        TraceRecord record = traceRecordRepository.findById(traceId)
            .orElseThrow(() -> new IllegalArgumentException("Trace record not found: " + traceId));
        record.linkChildBatch(childBatchNo, quantity);
        traceRecordRepository.update(record);
    }

    @Override
    public List<TraceRecord> forwardTrace(String traceCode) {
        List<TraceRecord> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        forwardDfs(traceCode, visited, result);
        return result;
    }

    @Override
    public List<TraceRecord> backwardTrace(String traceCode) {
        List<TraceRecord> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        backwardDfs(traceCode, visited, result);
        return result;
    }

    @Override
    public TraceRecord.TraceChain buildTraceChain(String traceCode) {
        TraceRecord root = traceRecordRepository.findByTraceCode(traceCode)
            .orElseThrow(() -> new IllegalArgumentException("Trace record not found: " + traceCode));
        TraceRecord.TraceChain chain = new TraceRecord.TraceChain();
        chain.setRoot(root);

        List<TraceRecord> forwardList = forwardTrace(traceCode);
        chain.setForwardPath(forwardList);

        List<TraceRecord> backwardList = backwardTrace(traceCode);
        chain.setBackwardPath(backwardList);

        return chain;
    }

    private void forwardDfs(String traceCode, Set<String> visited, List<TraceRecord> result) {
        if (traceCode == null || visited.contains(traceCode)) return;
        visited.add(traceCode);
        traceRecordRepository.findByTraceCode(traceCode).ifPresent(record -> {
            result.add(record);
            for (TraceRelation rel : record.getRelations()) {
                if ("SPLIT_FROM".equals(rel.getRelationType()) ||
                    "CONSUMED".equals(rel.getRelationType())) {
                    forwardDfs(rel.getRelatedCode(), visited, result);
                }
            }
        });
    }

    private void backwardDfs(String traceCode, Set<String> visited, List<TraceRecord> result) {
        if (traceCode == null || visited.contains(traceCode)) return;
        visited.add(traceCode);
        traceRecordRepository.findByTraceCode(traceCode).ifPresent(record -> {
            result.add(record);
            for (TraceRelation rel : record.getRelations()) {
                if ("PARENT".equals(rel.getRelationType())) {
                    backwardDfs(rel.getRelatedCode(), visited, result);
                }
            }
        });
    }
}
