package com.metawebthree.payment.domain.exception;

public class KycRequiredException extends PaymentException {
    public KycRequiredException() {
        super("KYC_REQUIRED", "KYC verification required");
    }
}