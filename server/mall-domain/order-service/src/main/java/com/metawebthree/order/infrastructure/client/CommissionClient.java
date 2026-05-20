package com.metawebthree.order.infrastructure.client;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.CalculateRequest;
import com.metawebthree.common.generated.rpc.CancelRequest;
import com.metawebthree.common.generated.rpc.CommissionService;
import com.metawebthree.order.domain.ports.CommissionSettlementPort;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

@Component
public class CommissionClient implements CommissionSettlementPort {

    @DubboReference(check = false)
    private CommissionService commissionService;

    @Override
    public void calculate(Long orderId, Long userId, BigDecimal payAmount, LocalDateTime availableAt) {
        if (orderId == null || userId == null || payAmount == null || availableAt == null) {
            throw new IllegalArgumentException("invalid commission calculation input");
        }
        
        CalculateRequest request = CalculateRequest.newBuilder()
                .setOrderId(orderId)
                .setUserId(userId)
                .setPayAmount(payAmount.toPlainString())
                .setAvailableAt(availableAt.toInstant(ZoneOffset.UTC).toEpochMilli())
                .build();
        
        commissionService.calculate(request);
    }

    @Override
    public void cancel(Long orderId) {
        if (orderId == null) {
            throw new IllegalArgumentException("invalid order id");
        }
        
        CancelRequest request = CancelRequest.newBuilder()
                .setOrderId(orderId)
                .build();
        
        commissionService.cancel(request);
    }
}
