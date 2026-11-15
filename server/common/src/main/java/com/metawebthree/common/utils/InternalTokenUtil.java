package com.metawebthree.common.utils;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.Base64;

/**
 * Signs and validates the short-lived gateway-to-service trust token.
 *
 * <p>Instances are created by {@code InternalTokenAutoConfiguration} from
 * {@code internal-token.*} properties, so the gateway and every downstream
 * service use the same secret and validity window. This class deliberately has
 * no stereotype annotation: a component-scanned bean would be configured
 * differently depending on which packages an application happens to scan.</p>
 */
public class InternalTokenUtil {

    private static final String HMAC_ALGORITHM = "HmacSHA256";

    private final String secret;
    private final long timeoutSeconds;

    public InternalTokenUtil(String secret, long timeoutSeconds) {
        this.secret = secret;
        this.timeoutSeconds = timeoutSeconds;
    }

    public String generate() {
        long now = Instant.now().getEpochSecond();
        return now + ":" + hmacHex(String.valueOf(now));
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
        } catch (NumberFormatException exception) {
            return false;
        }
        if (Math.abs(Instant.now().getEpochSecond() - timestamp) > timeoutSeconds) {
            return false;
        }
        byte[] expected = hmacHex(parts[0]).getBytes(StandardCharsets.UTF_8);
        byte[] actual = parts[1].getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(expected, actual);
    }

    private String hmacHex(String data) {
        try {
            Mac mac = Mac.getInstance(HMAC_ALGORITHM);
            mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), HMAC_ALGORITHM));
            return Base64.getEncoder().encodeToString(
                    mac.doFinal(data.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException | InvalidKeyException exception) {
            throw new IllegalStateException("HMAC initialization failed", exception);
        }
    }
}
