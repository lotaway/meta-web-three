package com.metawebthree.finance.domain.repository.cash;

import com.metawebthree.finance.domain.entity.cash.BankReconciliation;
import java.util.List;
import java.util.Optional;

public interface BankReconciliationRepository {
    BankReconciliation save(BankReconciliation reconciliation);
    Optional<BankReconciliation> findById(Long id);
    Optional<BankReconciliation> findByReconciliationNo(String reconciliationNo);
    List<BankReconciliation> findAll();
    List<BankReconciliation> findByBankAccountId(Long bankAccountId);
    List<BankReconciliation> findByStatus(BankReconciliation.ReconciliationStatus status);
    void deleteById(Long id);
}