package com.metawebthree.finance.domain.repository.ledger;

import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger.GeneralLedgerEntry;
import java.util.List;
import java.util.Optional;

public interface GeneralLedgerRepository {
    Optional<GeneralLedger> findById(Long id);
    Optional<GeneralLedger> findByLedgerNo(String ledgerNo);
    Optional<GeneralLedger> findByPeriod(Integer periodYear, Integer periodMonth);
    List<GeneralLedger> findByStatus(GeneralLedger.LedgerStatus status);
    List<GeneralLedger> findByPeriodBetween(Integer startYear, Integer startMonth, Integer endYear, Integer endMonth);
    List<GeneralLedgerEntry> findEntriesBySubjectId(Long subjectId);
    List<GeneralLedgerEntry> findEntriesBySubjectCode(String subjectCode);
    List<GeneralLedgerEntry> findEntriesByPeriod(Integer periodYear, Integer periodMonth);
    void save(GeneralLedger ledger);
    void update(GeneralLedger ledger);
    void delete(Long id);
}