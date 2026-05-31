package com.metawebthree.payment.domain.exception;

public class OrderNotFoundException extends PaymentException {
    public OrderNotFoundException(String orderNo) {
        super("ORDER_NOT_FOUND", "Order not found: " + orderNo);
    }

    public OrderNotFoundException(String orderNo, Throwable cause) {
        super("ORDER_NOT_FOUND", "Order not found: " + orderNo, cause);
    }
}