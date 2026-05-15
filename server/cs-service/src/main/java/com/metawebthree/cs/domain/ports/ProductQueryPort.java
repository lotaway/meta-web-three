package com.metawebthree.cs.domain.ports;

import java.util.Optional;

public interface ProductQueryPort {
    Optional<String> findProductName(Long productId);
    Optional<String> findProductJson(Long productId);
}
