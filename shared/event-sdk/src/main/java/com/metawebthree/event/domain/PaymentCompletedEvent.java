package com.metawebthree.event.domain;

import com.metawebthree.event.BaseEvent;
import com.metawebthree.event.EventType;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.math.BigDecimal;

/**
 * Event published when payment is completed.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
public class PaymentCompletedEvent extends BaseEvent {

    private String paymentId;
    private String orderId;
    private String userId;
    private BigDecimal amount;
    private String paymentMethod;
    private String transactionId;
}