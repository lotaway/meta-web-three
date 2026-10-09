package com.metawebthree.socialcommerce.infrastructure.client;

import java.math.BigDecimal;

import org.springframework.stereotype.Component;

import com.metawebthree.socialcommerce.domain.ports.CommissionPort;

import lombok.extern.slf4j.Slf4j;

/**
 * 分佣端口适配器（本地开发实现，仅记录日志）。
 */
@Slf4j
@Component
public class CommissionPortClient implements CommissionPort {

    @Override
    public void settleCommission(Long userId, BigDecimal amount, String source) {
        log.info("[stub] settleCommission userId={}, amount={}, source={}", userId, amount, source);
    }

    @Override
    public void freezeCommission(Long userId, BigDecimal amount) {
        log.info("[stub] freezeCommission userId={}, amount={}", userId, amount);
    }

    @Override
    public void unfreezeCommission(Long userId, BigDecimal amount) {
        log.info("[stub] unfreezeCommission userId={}, amount={}", userId, amount);
    }
}
