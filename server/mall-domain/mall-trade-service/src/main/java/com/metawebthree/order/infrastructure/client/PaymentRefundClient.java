package com.metawebthree.order.infrastructure.client;

import com.metawebthree.order.domain.ports.PaymentRefundPort;
import com.metawebthree.payment.application.PaymentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;

/**
 * Bridges the order domain refund flow to the payment channel via {@link PaymentService}.
 * The Alipay {@code out_trade_no} for an order is built as {@code ORDER_<orderId>}, matching
 * the value produced when the payment was created.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class PaymentRefundClient implements PaymentRefundPort {

    private final PaymentService paymentService;

    @Override
    public boolean refund(Long orderId, String orderNo, BigDecimal amount, String reason) {
        if (orderId == null) {
            throw new IllegalArgumentException("invalid order id");
        }
        String paymentOrderNo = "ORDER_" + orderId;
        String refundReason = reason != null && !reason.isBlank() ? reason : "Order refund";
        log.info("Requesting channel refund for orderId={}, paymentOrderNo={}, amount={}",
                orderId, paymentOrderNo, amount);
        return paymentService.refundPayment(paymentOrderNo, refundReason);
    }
}
