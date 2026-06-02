package com.metawebthree.finance.domain.repository.arap;

import com.metawebthree.finance.domain.entity.arap.AccountsReceivable;
import com.metawebthree.finance.domain.entity.arap.AccountsReceivable.ArStatus;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface AccountsReceivableRepository {
    Optional<AccountsReceivable> findById(Long id);
    Optional<AccountsReceivable> findByArCode(String arCode);
    List<AccountsReceivable> findByCustomerId(Long customerId);
    List<AccountsReceivable> findByStatus(ArStatus status);
    List<AccountsReceivable> findByDueDateBefore(LocalDate date);
    List<AccountsReceivable> findByCustomerIdAndStatus(Long customerId, ArStatus status);
    List<AccountsReceivable> findAll();
    AccountsReceivable save(AccountsReceivable ar);
    void delete(AccountsReceivable ar);
}