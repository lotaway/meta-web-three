package com.metawebthree.payment.domain.exception;

public class UnauthorizedAccessException extends PaymentException {
    public UnauthorizedAccessException(String message) {
        super("UNAUTHORIZED_ACCESS", message);
    }

    public UnauthorizedAccessException(String message, Throwable cause) {
        super("UNAUTHORIZED_ACCESS", message, cause);
    }
}