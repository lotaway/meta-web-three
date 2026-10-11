package com.metawebthree.payment.domain.exception;

import lombok.Getter;

@Getter
public class PaymentException extends RuntimeException {
    private final String code;
    private final String message;

    public PaymentException(String code, String message) {
        super(message);
        this.code = code;
        this.message = message;
    }

    public PaymentException(String code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.message = message;
    }
}