package com.metawebthree.payment.application.util;

import lombok.extern.slf4j.Slf4j;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Objects;

@Slf4j
public final class PaymentCallbackSignature {
    private static final String HMAC_ALGO = "HmacSHA256";

    private PaymentCallbackSignature() {
    }

    public static boolean verify(String secret, String paymentOrderNo, String status, String transactionId, String signature) {
        if (secret == null || secret.isBlank()) {
            log.error("Exchange payment callback secret not configured");
            return false;
        }
        if (paymentOrderNo == null || status == null) {
            log.error("Missing required callback parameters for signature verification");
            return false;
        }
        if (signature == null || signature.isBlank()) {
            log.error("Missing callback signature");
            return false;
        }
        String payload = buildPayload(paymentOrderNo, status, transactionId);
        String computed = signHex(secret, payload);
        boolean ok = MessageDigest.isEqual(computed.getBytes(StandardCharsets.UTF_8), signature.getBytes(StandardCharsets.UTF_8));
        if (!ok) {
            log.error("Exchange payment callback signature verification failed for order {}", paymentOrderNo);
        }
        return ok;
    }

    public static String signHex(String secret, String payload) {
        try {
            Mac mac = Mac.getInstance(HMAC_ALGO);
            mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), HMAC_ALGO));
            byte[] h = mac.doFinal(payload.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(h);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to compute HMAC signature", e);
        }
    }

    public static String buildPayload(String paymentOrderNo, String status, String transactionId) {
        String tx = transactionId == null ? "" : transactionId;
        return paymentOrderNo + "|" + status + "|" + tx;
    }
}
