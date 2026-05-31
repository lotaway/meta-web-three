package com.metawebthree.payment.domain.exception;

public class InvalidOrderStatusException extends PaymentException {
    public InvalidOrderStatusException(String status) {
        super("INVALID_ORDER_STATUS", "Cannot cancel order with status: " + status);
    }

    public InvalidOrderStatusException(String status, Throwable cause) {
        super("INVALID_ORDER_STATUS", "Cannot cancel order with status: " + status, cause);
    }
}