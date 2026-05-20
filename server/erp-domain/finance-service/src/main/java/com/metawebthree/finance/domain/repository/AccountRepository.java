package com.metawebthree.finance.domain.repository;

import com.metawebthree.finance.domain.entity.Account;
import java.util.List;
import java.util.Optional;

public interface AccountRepository {
    Optional<Account> findById(Long id);
    Optional<Account> findByAccountNo(String accountNo);
    List<Account> findByStatus(Account.AccountStatus status);
    List<Account> findByType(Account.AccountType type);
    List<Account> findAll();
    void save(Account account);
    void update(Account account);
    void delete(Long id);
}