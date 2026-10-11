package com.metawebthree.groupbuying.infrastructure.client;

import java.math.BigDecimal;

import org.springframework.stereotype.Component;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.groupbuying.domain.ports.OrderPort;

import lombok.extern.slf4j.Slf4j;

/**
 * 订单端口适配器（本地开发实现）。
 * 尚未接入 order-service 的真实 RPC，返回本地生成的订单 ID，保证服务可独立启动。
 */
@Slf4j
@Component
public class OrderPortClient implements OrderPort {

    @Override
    public Long createOrder(Long userId, Long productId, Integer quantity, BigDecimal amount, String remark) {
        Long orderId = IdWorker.getId();
        log.info("[stub] createOrder userId={}, productId={}, quantity={}, amount={}, remark={} -> orderId={}",
                userId, productId, quantity, amount, remark, orderId);
        return orderId;
    }

    @Override
    public void cancelOrder(Long orderId) {
        log.info("[stub] cancelOrder orderId={}", orderId);
    }

    @Override
    public void payOrder(Long orderId) {
        log.info("[stub] payOrder orderId={}", orderId);
    }
}
