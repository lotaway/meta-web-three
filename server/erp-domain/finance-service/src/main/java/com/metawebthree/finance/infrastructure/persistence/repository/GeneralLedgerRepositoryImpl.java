package com.metawebthree.finance.infrastructure.persistence.repository;

import com.metawebthree.finance.domain.entity.ledger.GeneralLedger;
import com.metawebthree.finance.domain.entity.ledger.GeneralLedger.GeneralLedgerEntry;
import com.metawebthree.finance.domain.repository.ledger.GeneralLedgerRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * In-memory implementation of GeneralLedgerRepository.
 * There is no general_ledger table in schema.sql yet; this keeps the ledger
 * domain service functional until a persistent mapper is introduced.
 */
@Repository
public class GeneralLedgerRepositoryImpl implements GeneralLedgerRepository {

    private final ConcurrentMap<Long, GeneralLedger> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public Optional<GeneralLedger> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<GeneralLedger> findByLedgerNo(String ledgerNo) {
        return store.values().stream()
                .filter(l -> l.getLedgerNo() != null && l.getLedgerNo().equals(ledgerNo))
                .findFirst();
    }

    @Override
    public Optional<GeneralLedger> findByPeriod(Integer periodYear, Integer periodMonth) {
        return store.values().stream()
                .filter(l -> periodYear.equals(l.getPeriodYear()) && periodMonth.equals(l.getPeriodMonth()))
                .findFirst();
    }

    @Override
    public List<GeneralLedger> findByStatus(GeneralLedger.LedgerStatus status) {
        return store.values().stream()
                .filter(l -> l.getStatus() == status)
                .collect(Collectors.toList());
    }

    @Override
    public List<GeneralLedger> findByPeriodBetween(Integer startYear, Integer startMonth, Integer endYear, Integer endMonth) {
        return store.values().stream()
                .filter(l -> {
                    int y = l.getPeriodYear();
                    int m = l.getPeriodMonth();
                    return (y > startYear || (y == startYear && m >= startMonth))
                            && (y < endYear || (y == endYear && m <= endMonth));
                })
                .collect(Collectors.toList());
    }

    @Override
    public List<GeneralLedgerEntry> findEntriesBySubjectId(Long subjectId) {
        return store.values().stream()
                .flatMap(l -> l.getEntries().stream())
                .filter(e -> subjectId.equals(e.getSubjectId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<GeneralLedgerEntry> findEntriesBySubjectCode(String subjectCode) {
        return store.values().stream()
                .flatMap(l -> l.getEntries().stream())
                .filter(e -> e.getSubjectCode() != null && e.getSubjectCode().equals(subjectCode))
                .collect(Collectors.toList());
    }

    @Override
    public List<GeneralLedgerEntry> findEntriesByPeriod(Integer periodYear, Integer periodMonth) {
        return store.values().stream()
                .filter(l -> periodYear.equals(l.getPeriodYear()) && periodMonth.equals(l.getPeriodMonth()))
                .flatMap(l -> l.getEntries().stream())
                .collect(Collectors.toList());
    }

    @Override
    public void save(GeneralLedger ledger) {
        if (ledger.getId() == null) {
            ledger.setId(idSeq.getAndIncrement());
        }
        store.put(ledger.getId(), ledger);
    }

    @Override
    public void update(GeneralLedger ledger) {
        if (ledger.getId() == null) {
            save(ledger);
            return;
        }
        store.put(ledger.getId(), ledger);
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}