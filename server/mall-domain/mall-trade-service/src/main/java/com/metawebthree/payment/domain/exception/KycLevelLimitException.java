package com.metawebthree.payment.domain.exception;

public class KycLevelLimitException extends PaymentException {
    public KycLevelLimitException(String levelDescription, int limit) {
        super("KYC_LEVEL_LIMIT_EXCEEDED", "Amount exceeds KYC level limit. Current level: " + levelDescription + ", Limit: " + limit + " USD");
    }
}