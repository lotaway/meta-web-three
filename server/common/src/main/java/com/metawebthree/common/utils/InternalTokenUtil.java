package com.metawebthree.common.utils;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.Base64;

@Component
public class InternalTokenUtil {

    private final String secret;
    private final long timeoutSeconds;

    private static final String HMAC_ALGORITHM = "HmacSHA256";

    public InternalTokenUtil(
            @Value("${internal-token.secret:mwt-internal-secret-2024}") String secret,
            @Value("${internal-token.timeout-seconds:10}") long timeoutSeconds) {
        this.secret = secret;
        this.timeoutSeconds = timeoutSeconds;
    }

    public String generate() {
        long now = Instant.now().getEpochSecond();
        String signature = hmacHex(String.valueOf(now));
        return now + ":" + signature;
    }

    public boolean validate(String token) {
        if (token == null || token.isBlank()) {
            return false;
        }
        String[] parts = token.split(":", 2);
        if (parts.length != 2) {
            return false;
        }
        long timestamp;
        try {
            timestamp = Long.parseLong(parts[0]);
        } catch (NumberFormatException e) {
            return false;
        }
        long now = Instant.now().getEpochSecond();
        if (Math.abs(now - timestamp) > timeoutSeconds) {
            return false;
        }
        String expected = hmacHex(parts[0]);
        return expected.equals(parts[1]);
    }

    private String hmacHex(String data) {
        try {
            Mac mac = Mac.getInstance(HMAC_ALGORITHM);
            mac.init(new SecretKeySpec(secret.getBytes(), HMAC_ALGORITHM));
            byte[] hash = mac.doFinal(data.getBytes());
            return Base64.getEncoder().encodeToString(hash);
        } catch (NoSuchAlgorithmException | InvalidKeyException e) {
            throw new RuntimeException("HMAC initialization failed", e);
        }
    }
}
