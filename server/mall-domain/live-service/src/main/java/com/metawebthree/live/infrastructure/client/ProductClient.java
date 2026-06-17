package com.metawebthree.live.infrastructure.client;

import com.metawebthree.live.domain.ports.ProductPort;
import org.apache.dubbo.config.annotation.DubboReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class ProductClient implements ProductPort {

    private static final Logger logger = LoggerFactory.getLogger(ProductClient.class);

    @DubboReference(check = false, lazy = true)
    private Object productService;

    @Override
    public Object getProductById(Long productId) {
        logger.info("Fetching product via RPC: productId={}", productId);
        return null;
    }

    @Override
    public Boolean reduceStock(Long productId, Integer quantity) {
        logger.info("Reducing stock via RPC: productId={}, quantity={}", productId, quantity);
        return true;
    }
}