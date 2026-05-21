package com.metawebthree.commission.domain.ports;

import java.time.LocalDateTime;

import com.metawebthree.commission.domain.CommissionAccount;

public interface CommissionAccountStore {
    CommissionAccount findByUserId(Long userId);
    void save(CommissionAccount account);
    void updateBalances(Long id, CommissionAccount updated, LocalDateTime updatedAt);
}
