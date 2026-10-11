package com.metawebthree.socialcommerce.domain.ports;

import java.math.BigDecimal;

public interface CommissionPort {
    void settleCommission(Long userId, BigDecimal amount, String source);
    void freezeCommission(Long userId, BigDecimal amount);
    void unfreezeCommission(Long userId, BigDecimal amount);
}