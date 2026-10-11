package com.metawebthree.settlement.domain.repository;

import com.metawebthree.settlement.domain.entity.ReconciliationRecord;
import java.util.List;
import java.util.Optional;

public interface ReconciliationRecordRepository {
    Optional<ReconciliationRecord> findById(Long id);
    Optional<ReconciliationRecord> findByRecordNo(String recordNo);
    List<ReconciliationRecord> findByStatus(ReconciliationRecord.ReconciliationStatus status);
    List<ReconciliationRecord> findByReconcileDateBetween(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<ReconciliationRecord> findAll();
    void save(ReconciliationRecord record);
    void update(ReconciliationRecord record);
    void delete(Long id);
}