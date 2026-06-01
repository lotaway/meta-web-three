package com.metawebthree.live.domain.ports;

public interface OrderPort {
    Long createOrder(Long userId, Long productId, Integer quantity, Long roomId);
}