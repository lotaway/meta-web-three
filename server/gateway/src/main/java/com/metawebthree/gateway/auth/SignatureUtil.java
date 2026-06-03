package com.metawebthree.gateway.auth;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;

/**
 * Utility class for API request signature generation and verification
 * Prevents request tampering and ensures data integrity
 */
public class SignatureUtil {
    
    private static final String HMAC_SHA256 = "HmacSHA256";
    private static final String MD5 = "MD5";
    
    /**
     * Generate signature for API request
     * 
     * @param method HTTP method (GET, POST, etc.)
     * @param path Request path
     * @param params Request parameters (query + body)
     * @param timestamp Request timestamp
     * @param nonce Random nonce string
     * @param apiSecret API secret key
     * @return Generated signature
     */
    public static String generateSignature(String method, String path, 
                                         Map<String, String> params, 
                                         long timestamp, String nonce, 
                                         String apiSecret) {
        try {
            // 1. Sort parameters lexicographically
            Map<String, String> sortedParams = new TreeMap<>(params);
            
            // 2. Build parameter string
            StringBuilder paramStr = new StringBuilder();
            for (Map.Entry<String, String> entry : sortedParams.entrySet()) {
                if (paramStr.length() > 0) {
                    paramStr.append("&");
                }
                paramStr.append(entry.getKey()).append("=").append(entry.getValue());
            }
            
            // 3. Build signature string: method + path + paramStr + timestamp + nonce
            String signString = method + path + paramStr.toString() + timestamp + nonce;
            
            // 4. Generate HMAC-SHA256 signature
            Mac mac = Mac.getInstance(HMAC_SHA256);
            SecretKeySpec secretKeySpec = new SecretKeySpec(
                apiSecret.getBytes(StandardCharsets.UTF_8), HMAC_SHA256);
            mac.init(secretKeySpec);
            
            byte[] hash = mac.doFinal(signString.getBytes(StandardCharsets.UTF_8));
            
            // 5. Convert to hex string
            return bytesToHex(hash).toLowerCase();
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate signature", e);
        }
    }
    
    /**
     * Verify API request signature
     * 
     * @param method HTTP method
     * @param path Request path
     * @param params Request parameters
     * @param timestamp Request timestamp
     * @param nonce Random nonce string
     * @param apiSecret API secret key
     * @param clientSignature Signature from client request
     * @return true if signature is valid
     */
    public static boolean verifySignature(String method, String path,
                                        Map<String, String> params,
                                        long timestamp, String nonce,
                                        String apiSecret, String clientSignature) {
        // 1. Check timestamp (prevent replay attacks - within 5 minutes)
        long currentTime = System.currentTimeMillis() / 1000;
        if (Math.abs(currentTime - timestamp) > 300) {
            return false;
        }
        
        // 2. Generate expected signature
        String expectedSignature = generateSignature(method, path, params, 
                                                    timestamp, nonce, apiSecret);
        
        // 3. Compare signatures (constant-time comparison to prevent timing attacks)
        return constantTimeEquals(expectedSignature, clientSignature);
    }
    
    /**
     * Generate MD5 hash (used for parameter signing in some scenarios)
     */
    public static String md5(String data) {
        try {
            MessageDigest md = MessageDigest.getInstance(MD5);
            byte[] hash = md.digest(data.getBytes(StandardCharsets.UTF_8));
            return bytesToHex(hash).toLowerCase();
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate MD5", e);
        }
    }
    
    /**
     * Convert byte array to hex string
     */
    private static String bytesToHex(byte[] bytes) {
        StringBuilder hexString = new StringBuilder();
        for (byte b : bytes) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) {
                hexString.append('0');
            }
            hexString.append(hex);
        }
        return hexString.toString();
    }
    
    /**
     * Constant-time string comparison to prevent timing attacks
     */
    private static boolean constantTimeEquals(String a, String b) {
        if (a == null || b == null) {
            return false;
        }
        
        if (a.length() != b.length()) {
            return false;
        }
        
        int result = 0;
        for (int i = 0; i < a.length(); i++) {
            result |= a.charAt(i) ^ b.charAt(i);
        }
        return result == 0;
    }
    
    /**
     * Generate random nonce string
     */
    public static String generateNonce() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }
}