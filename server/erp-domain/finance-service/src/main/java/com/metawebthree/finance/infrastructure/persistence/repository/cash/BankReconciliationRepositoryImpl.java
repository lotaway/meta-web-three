package com.metawebthree.finance.infrastructure.persistence.repository.cash;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cash.BankReconciliation;
import com.metawebthree.finance.domain.repository.cash.BankReconciliationRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 银行对账仓储（本地开发用内存实现）。
 */
@Repository
public class BankReconciliationRepositoryImpl implements BankReconciliationRepository {

    private final Map<Long, BankReconciliation> store = new ConcurrentHashMap<>();

    @Override
    public Long save(BankReconciliation reconciliation) {
        if (reconciliation.getId() == null) {
            reconciliation.setId(IdWorker.getId());
        }
        store.put(reconciliation.getId(), reconciliation);
        return reconciliation.getId();
    }

    @Override
    public Optional<BankReconciliation> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<BankReconciliation> findByReconciliationNo(String reconciliationNo) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getReconciliationNo(), reconciliationNo))
                .findFirst();
    }

    @Override
    public List<BankReconciliation> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<BankReconciliation> findByBankAccountId(Long bankAccountId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getBankAccountId(), bankAccountId))
                .collect(Collectors.toList());
    }

    @Override
    public List<BankReconciliation> findByStatus(BankReconciliation.ReconciliationStatus status) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public void update(BankReconciliation reconciliation) {
        if (reconciliation.getId() != null) {
            store.put(reconciliation.getId(), reconciliation);
        }
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}
