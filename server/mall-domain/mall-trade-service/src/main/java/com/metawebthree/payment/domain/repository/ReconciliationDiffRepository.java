package com.metawebthree.payment.domain.repository;

import com.metawebthree.payment.infrastructure.persistence.dataobject.ReconciliationDiffDO;

import java.time.LocalDate;
import java.util.List;

public interface ReconciliationDiffRepository {

    /**
     * Save reconciliation difference record
     */
    void save(ReconciliationDiffDO diff);

    /**
     * Batch save reconciliation difference records
     */
    void saveBatch(List<ReconciliationDiffDO> diffs);

    /**
     * Find differences by reconciliation date
     */
    List<ReconciliationDiffDO> findByReconciliationDate(LocalDate reconciliationDate);

    /**
     * Find pending differences by reconciliation date
     */
    List<ReconciliationDiffDO> findPendingByReconciliationDate(LocalDate reconciliationDate);

    /**
     * Count differences by type and date
     */
    Long countByReconciliationDateAndDiffType(LocalDate reconciliationDate, String diffType);
}