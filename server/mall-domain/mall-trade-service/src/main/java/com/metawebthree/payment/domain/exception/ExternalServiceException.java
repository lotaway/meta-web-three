package com.metawebthree.payment.domain.exception;

public class ExternalServiceException extends PaymentException {
    public ExternalServiceException(String serviceName, String message) {
        super("EXTERNAL_SERVICE_ERROR", serviceName + ": " + message);
    }

    public ExternalServiceException(String serviceName, String message, Throwable cause) {
        super("EXTERNAL_SERVICE_ERROR", serviceName + ": " + message, cause);
    }
}