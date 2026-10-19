package com.metawebthree.order.domain.ports;

import java.math.BigDecimal;

/**
 * Port for requesting a refund from the external payment channel for an order.
 * Implementations must be idempotent at the channel level where the provider supports it.
 */
public interface PaymentRefundPort {

    /**
     * Request a refund for the given order.
     *
     * @param orderId the order id
     * @param orderNo the business order number
     * @param amount  the amount to refund
     * @param reason  a human readable refund reason
     * @return true when the payment channel accepted the refund, false when it explicitly failed
     */
    boolean refund(Long orderId, String orderNo, BigDecimal amount, String reason);
}
