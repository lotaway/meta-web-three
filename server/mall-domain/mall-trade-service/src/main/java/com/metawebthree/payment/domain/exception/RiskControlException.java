package com.metawebthree.payment.domain.exception;

public class RiskControlException extends PaymentException {
    public RiskControlException(String message) {
        super("RISK_CONTROL_FAILED", message);
    }

    public RiskControlException(String message, Throwable cause) {
        super("RISK_CONTROL_FAILED", message, cause);
    }
}