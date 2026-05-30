package com.metawebthree.finance.domain.repository.cash;

import com.metawebthree.finance.domain.entity.cash.BankAccount;
import java.util.List;
import java.util.Optional;

public interface BankAccountRepository {
    Long save(BankAccount bankAccount);
    void update(BankAccount bankAccount);
    Optional<BankAccount> findById(Long id);
    Optional<BankAccount> findByAccountCode(String accountCode);
    List<BankAccount> findAll();
    List<BankAccount> findByStatus(BankAccount.BankAccountStatus status);
    List<BankAccount> findByIsActive(Boolean isActive);
    void deleteById(Long id);
}