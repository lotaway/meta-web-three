package com.metawebthree.order.infrastructure.client;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import com.metawebthree.order.domain.ports.CommissionSettlementPort;

@Component
public class CommissionClient implements CommissionSettlementPort {
    private final RestTemplate restTemplate;
    private final String baseUrl;

    public CommissionClient(RestTemplateBuilder builder,
            @Value("${commission.service.base-url}") String baseUrl) {
        this.restTemplate = builder.build();
        this.baseUrl = baseUrl;
    }

    @Override
    public void calculate(Long orderId, Long userId, BigDecimal payAmount, LocalDateTime availableAt) {
        if (orderId == null || userId == null || payAmount == null || availableAt == null) {
            throw new IllegalArgumentException("invalid commission calculation input");
        }
        CalcRequest request = new CalcRequest(orderId, userId, payAmount, availableAt);
        restTemplate.postForEntity(baseUrl + "/v1/commission/calc", request, Void.class);
    }

    @Override
    public void cancel(Long orderId) {
        if (orderId == null) {
            throw new IllegalArgumentException("invalid order id");
        }
        CancelRequest request = new CancelRequest(orderId);
        restTemplate.postForEntity(baseUrl + "/v1/commission/cancel", request, Void.class);
    }

    private static class CalcRequest {
        private Long orderId;
        private Long userId;
        private BigDecimal payAmount;
        private LocalDateTime availableAt;

        CalcRequest(Long orderId, Long userId, BigDecimal payAmount, LocalDateTime availableAt) {
            this.orderId = orderId;
            this.userId = userId;
            this.payAmount = payAmount;
            this.availableAt = availableAt;
        }

        public Long getOrderId() { return orderId; }
        public void setOrderId(Long orderId) { this.orderId = orderId; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public BigDecimal getPayAmount() { return payAmount; }
        public void setPayAmount(BigDecimal payAmount) { this.payAmount = payAmount; }
        public LocalDateTime getAvailableAt() { return availableAt; }
        public void setAvailableAt(LocalDateTime availableAt) { this.availableAt = availableAt; }
    }

    private static class CancelRequest {
        private Long orderId;

        CancelRequest(Long orderId) {
            this.orderId = orderId;
        }

        public Long getOrderId() { return orderId; }
        public void setOrderId(Long orderId) { this.orderId = orderId; }
    }
}
