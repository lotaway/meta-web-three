package com.metawebthree.live.domain.ports;

public interface ProductPort {
    Object getProductById(Long productId);
    Boolean reduceStock(Long productId, Integer quantity);
}