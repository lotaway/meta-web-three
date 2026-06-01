package com.metawebthree.groupbuying.domain.ports;

import java.math.BigDecimal;

public interface OrderPort {
    Long createOrder(Long userId, Long productId, Integer quantity, BigDecimal amount, String remark);
    void cancelOrder(Long orderId);
    void payOrder(Long orderId);
}