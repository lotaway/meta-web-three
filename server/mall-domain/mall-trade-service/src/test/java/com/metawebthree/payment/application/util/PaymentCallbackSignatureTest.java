package com.metawebthree.payment.application.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PaymentCallbackSignatureTest {

    private static final String SECRET = "exchange-callback-secret";

    @Test
    void acceptsValidSignature() {
        String signature = sign("PAY-1", "SUCCESS", "TX-1");
        assertTrue(PaymentCallbackSignature.verify(SECRET, "PAY-1", "SUCCESS", "TX-1", signature));
    }

    @Test
    void rejectsTamperedStatus() {
        String signature = sign("PAY-1", "SUCCESS", "TX-1");
        assertFalse(PaymentCallbackSignature.verify(SECRET, "PAY-1", "FAILED", "TX-1", signature));
    }

    @Test
    void rejectsTamperedTransactionId() {
        String signature = sign("PAY-1", "SUCCESS", "TX-1");
        assertFalse(PaymentCallbackSignature.verify(SECRET, "PAY-1", "SUCCESS", "TX-2", signature));
    }

    @Test
    void rejectsTamperedPaymentOrderNo() {
        String signature = sign("PAY-1", "SUCCESS", "TX-1");
        assertFalse(PaymentCallbackSignature.verify(SECRET, "PAY-2", "SUCCESS", "TX-1", signature));
    }

    @Test
    void rejectsBlankSecret() {
        String signature = sign("PAY-1", "SUCCESS", "TX-1");
        assertFalse(PaymentCallbackSignature.verify("", "PAY-1", "SUCCESS", "TX-1", signature));
    }

    @Test
    void rejectsBlankSignature() {
        assertFalse(PaymentCallbackSignature.verify(SECRET, "PAY-1", "SUCCESS", "TX-1", ""));
        assertFalse(PaymentCallbackSignature.verify(SECRET, "PAY-1", "SUCCESS", "TX-1", null));
    }

    private String sign(String paymentOrderNo, String status, String transactionId) {
        String payload = PaymentCallbackSignature.buildPayload(paymentOrderNo, status, transactionId);
        return PaymentCallbackSignature.signHex(SECRET, payload);
    }
}
