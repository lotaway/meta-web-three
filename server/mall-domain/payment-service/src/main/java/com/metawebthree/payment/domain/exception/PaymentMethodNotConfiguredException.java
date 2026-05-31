package com.metawebthree.payment.domain.exception;

/**
 * Exception thrown when payment method is not configured
 */
public class PaymentMethodNotConfiguredException extends RuntimeException {

    private final String paymentMethod;

    public PaymentMethodNotConfiguredException(String paymentMethod) {
        super(paymentMethod + " payment method is not configured");
        this.paymentMethod = paymentMethod;
    }

    public PaymentMethodNotConfiguredException(String paymentMethod, String message) {
        super(message);
        this.paymentMethod = paymentMethod;
    }

    public String getPaymentMethod() {
        return paymentMethod;
    }
}