package com.metawebthree.live.infrastructure.client;

import com.metawebthree.live.domain.ports.OrderPort;
import org.apache.dubbo.config.annotation.DubboReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class OrderClient implements OrderPort {

    private static final Logger logger = LoggerFactory.getLogger(OrderClient.class);

    @DubboReference(check = false, lazy = true)
    private Object orderService;

    @Override
    public Long createOrder(Long userId, Long productId, Integer quantity, Long roomId) {
        logger.info("Creating order via RPC: userId={}, productId={}, quantity={}, roomId={}", 
                userId, productId, quantity, roomId);
        return System.currentTimeMillis();
    }
}