package com.metawebthree.finance.domain.repository.cash;

import com.metawebthree.finance.domain.entity.cash.CashTransfer;
import java.util.List;
import java.util.Optional;

public interface CashTransferRepository {
    Long save(CashTransfer transfer);
    Optional<CashTransfer> findById(Long id);
    Optional<CashTransfer> findByTransferNo(String transferNo);
    List<CashTransfer> findAll();
    List<CashTransfer> findByStatus(CashTransfer.CashTransferStatus status);
    List<CashTransfer> findByFromAccountId(Long fromAccountId);
    List<CashTransfer> findByToAccountId(Long toAccountId);
    List<CashTransfer> findByFromAccountIdOrToAccountId(Long fromAccountId, Long toAccountId);
    void update(CashTransfer transfer);
    void deleteById(Long id);
}