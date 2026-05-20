package com.metawebthree.event;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;

/**
 * Event published when a new order is created.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OrderCreatedEvent extends BaseEvent {

    private String orderId;
    private String userId;
    private BigDecimal totalAmount;
    private String currency;
    private List<OrderItem> items;
    private String shippingAddress;
    private Instant orderTime;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class OrderItem {
        private String productId;
        private String productName;
        private Integer quantity;
        private BigDecimal unitPrice;
    }
}