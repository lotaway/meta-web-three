package com.metawebthree.finance.domain.repository.arap;

import com.metawebthree.finance.domain.entity.arap.AccountsPayable;
import com.metawebthree.finance.domain.entity.arap.AccountsPayable.ApStatus;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface AccountsPayableRepository {
    Optional<AccountsPayable> findById(Long id);
    Optional<AccountsPayable> findByApCode(String apCode);
    List<AccountsPayable> findBySupplierId(Long supplierId);
    List<AccountsPayable> findByStatus(ApStatus status);
    List<AccountsPayable> findByDueDateBefore(LocalDate date);
    List<AccountsPayable> findBySupplierIdAndStatus(Long supplierId, ApStatus status);
    List<AccountsPayable> findAll();
    AccountsPayable save(AccountsPayable ap);
    void delete(AccountsPayable ap);
}