package com.metawebthree.order.domain.ports;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public interface CommissionSettlementPort {
    void calculate(Long orderId, Long userId, BigDecimal payAmount, LocalDateTime availableAt);
    void cancel(Long orderId);
}
