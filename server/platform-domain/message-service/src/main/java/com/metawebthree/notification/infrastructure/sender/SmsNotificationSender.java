package com.metawebthree.notification.infrastructure.sender;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import com.metawebthree.notification.domain.ports.UserQueryPort;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * SMS notification sender implementation
 * Integrates with SMS gateway to send text messages
 */
@Component
public class SmsNotificationSender implements NotificationSender {
    
    private static final Logger logger = LoggerFactory.getLogger(SmsNotificationSender.class);
    private static final String CHANNEL_CODE = "SMS";
    
    @Autowired
    private UserQueryPort userQueryPort;
    
    @Autowired
    private RestTemplate restTemplate;
    
    // SMS gateway configuration
    @Value("${notification.sms.gateway.url:}")
    private String gatewayUrl;
    
    @Value("${notification.sms.gateway.appkey:}")
    private String appKey;
    
    @Value("${notification.sms.gateway.secret:}")
    private String appSecret;
    
    @Value("${notification.sms.enabled:true}")
    private boolean enabled;
    
    // Simulated SMS sending records for demonstration
    private final ConcurrentHashMap<Long, SmsRecord> smsRecordStore = new ConcurrentHashMap<>();
    
    @Override
    public boolean send(Long userId, String title, String content, String extraData) {
        if (!enabled) {
            logger.warn("SMS notification is disabled");
            return false;
        }
        
        try {
            // Query phone number from user service
            String phone = getPhoneByUserId(userId);
            if (phone == null) {
                logger.warn("Cannot send SMS: phone number not found for userId: {}", userId);
                return false;
            }
            
            // Build SMS request
            SmsRequest request = buildSmsRequest(userId, title, content, extraData, phone);
            
            // Call SMS gateway (simulated in this implementation)
            SmsResponse response = invokeSmsGateway(request);
            
            if (response.isSuccess()) {
                // Record SMS for tracking
                SmsRecord record = new SmsRecord();
                record.setUserId(userId);
                record.setPhone(response.getPhone());
                record.setMessage(content);
                record.setMessageId(response.getMessageId());
                record.setSendTime(System.currentTimeMillis());
                record.setStatus("SENT");
                smsRecordStore.put(userId, record);
                
                logger.info("SMS notification sent successfully to user: {}, messageId: {}", userId, response.getMessageId());
                return true;
            } else {
                logger.error("SMS gateway returned error: {}", response.getErrorMessage());
                return false;
            }
        } catch (Exception e) {
            logger.error("Failed to send SMS notification to user: {}, error: {}", userId, e.getMessage());
            return false;
        }
    }
    
    private SmsRequest buildSmsRequest(Long userId, String title, String content, String extraData, String phone) {
        SmsRequest request = new SmsRequest();
        request.setAppKey(appKey);
        request.setAppSecret(appSecret);
        request.setUserId(userId);
        request.setPhone(phone);
        request.setMessage(buildSmsMessage(title, content));
        request.setTimestamp(System.currentTimeMillis());
        return request;
    }
    
    private String buildSmsMessage(String title, String content) {
        // Truncate content for SMS (max 70 Chinese characters or 160 English characters)
        String fullMessage = title + ": " + content;
        if (fullMessage.length() > 70) {
            fullMessage = fullMessage.substring(0, 67) + "...";
        }
        return fullMessage;
    }
    
    private String getPhoneByUserId(Long userId) {
        // Query real phone number from user service via UserQueryPort
        return userQueryPort.findPhone(userId).orElseGet(() -> {
            logger.warn("Unable to query phone for userId: {}, SMS notification will not be sent", userId);
            return null;
        });
    }
    
    private SmsResponse invokeSmsGateway(SmsRequest request) {
        SmsResponse response = new SmsResponse();
        
        if (gatewayUrl == null || gatewayUrl.isEmpty()) {
            logger.warn("SMS gateway URL not configured, falling back to simulated response");
            response.setSuccess(true);
            response.setMessageId("SMS_" + System.currentTimeMillis());
            response.setPhone(request.getPhone());
            return response;
        }
        
        try {
            // Build HTTP request headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            // Build payload for SMS gateway
            Map<String, Object> payload = new HashMap<>();
            payload.put("phone", request.getPhone());
            payload.put("message", request.getMessage());
            payload.put("appKey", request.getAppKey());
            payload.put("timestamp", request.getTimestamp());
            payload.put("sign", generateSign(request));
            
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
            
            // Call SMS gateway API
            ResponseEntity<Map> responseEntity = restTemplate.postForEntity(
                    gatewayUrl + "/send",
                    entity,
                    Map.class
            );
            
            if (responseEntity.getStatusCode() == HttpStatus.OK && responseEntity.getBody() != null) {
                Map<String, Object> responseBody = responseEntity.getBody();
                Object code = responseBody.get("code");
                
                if ("0".equals(String.valueOf(code)) || "200".equals(String.valueOf(code))) {
                    response.setSuccess(true);
                    response.setMessageId(String.valueOf(responseBody.getOrDefault("messageId", "SMS_" + System.currentTimeMillis())));
                    response.setPhone(request.getPhone());
                    logger.info("SMS sent successfully via gateway to phone: ***", 
                            request.getPhone() != null && request.getPhone().length() > 7 ? 
                                    request.getPhone().substring(request.getPhone().length() - 4) : "unknown");
                } else {
                    response.setSuccess(false);
                    response.setErrorMessage(String.valueOf(responseBody.getOrDefault("message", "Unknown error")));
                    logger.error("SMS gateway returned error: code={}, message={}", 
                            code, response.getErrorMessage());
                }
            } else {
                response.setSuccess(false);
                response.setErrorMessage("HTTP request failed: " + responseEntity.getStatusCode());
                logger.error("SMS gateway HTTP error: {}", responseEntity.getStatusCode());
            }
        } catch (Exception e) {
            logger.error("Failed to invoke SMS gateway: {}", e.getMessage(), e);
            response.setSuccess(false);
            response.setErrorMessage("SMS gateway invocation failed: " + e.getMessage());
        }
        
        return response;
    }
    
    /**
     * Generate signature for SMS API request
     */
    private String generateSign(SmsRequest request) {
        // Simple signature generation: MD5(appKey + appSecret + timestamp)
        // In production, use proper HMAC or other signature algorithm
        String signSource = request.getAppKey() + request.getAppSecret() + request.getTimestamp();
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(signSource.getBytes());
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            logger.warn("Failed to generate sign, using empty string", e);
            return "";
        }
    }
    
    @Override
    public String getChannelCode() {
        return CHANNEL_CODE;
    }
    
    // Internal classes for SMS communication
    @Data
    private static class SmsRequest {
        private String appKey;
        private String appSecret;
        private Long userId;
        private String phone;
        private String message;
        private long timestamp;
    }
    
    @Data
    private static class SmsResponse {
        private boolean success;
        private String messageId;
        private String phone;
        private String errorMessage;
    }
    
    @Data
    private static class SmsRecord {
        private Long userId;
        private String phone;
        private String message;
        private String messageId;
        private long sendTime;
        private String status;
    }
}