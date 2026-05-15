package com.metawebthree.cs.domain.ports;

import java.util.Optional;

public interface OrderQueryPort {
    Optional<String> findOrderStatus(Long orderId);
    Optional<String> findOrderJson(Long orderId);
}
