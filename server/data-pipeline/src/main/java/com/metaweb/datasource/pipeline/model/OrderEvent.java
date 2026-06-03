package com.metaweb.datasource.pipeline.model;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class OrderEvent {
    private String eventId;
    private String eventType;
    private Long orderId;
    private Long userId;
    private BigDecimal totalAmount;
    private String status;
    private LocalDateTime eventTime;
    private String productInfo;
    private String paymentMethod;
    private Long merchantId;

    public OrderEvent() {
    }

    public OrderEvent(String eventId, String eventType, Long orderId, Long userId) {
        this.eventId = eventId;
        this.eventType = eventType;
        this.orderId = orderId;
        this.userId = userId;
        this.eventTime = LocalDateTime.now();
    }
}
