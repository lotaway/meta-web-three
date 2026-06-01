package com.metawebthree.notification.infrastructure.sender;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * App push notification sender implementation
 * Integrates with push notification services (FCM, JPush, etc.)
 */
@Component
public class AppPushNotificationSender implements NotificationSender {
    
    private static final Logger logger = LoggerFactory.getLogger(AppPushNotificationSender.class);
    private static final String CHANNEL_CODE = "APP";
    
    // Push notification service configuration
    @Value("${notification.push.enabled:true}")
    private boolean enabled;
    
    @Value("${notification.push.provider:fcm}")
    private String provider; // fcm, jpush,华为push等
    
    @Value("${notification.push.fcm.project-id:}")
    private String fcmProjectId;
    
    @Value("${notification.push.jpush.app-key:}")
    private String jpushAppKey;
    
    @Value("${notification.push.jpush.master-secret:}")
    private String jpushMasterSecret;
    
    // Push notification records
    private final ConcurrentHashMap<Long, PushRecord> pushRecordStore = new ConcurrentHashMap<>();
    
    @Override
    public boolean send(Long userId, String title, String content, String extraData) {
        if (!enabled) {
            logger.warn("App push notification is disabled");
            return false;
        }
        
        try {
            // Get user device token
            String deviceToken = getDeviceTokenByUserId(userId);
            
            if (deviceToken == null || deviceToken.isEmpty()) {
                logger.error("Cannot send push notification: no device token for user: {}", userId);
                return false;
            }
            
            // Build push request
            PushRequest request = buildPushRequest(userId, deviceToken, title, content, extraData);
            
            // Send push notification based on provider
            PushResponse response = sendPushNotification(request);
            
            if (response.isSuccess()) {
                // Record for tracking
                PushRecord record = new PushRecord();
                record.setUserId(userId);
                record.setDeviceToken(deviceToken);
                record.setTitle(title);
                record.setMessageId(response.getMessageId());
                record.setSendTime(System.currentTimeMillis());
                record.setStatus("SENT");
                pushRecordStore.put(userId, record);
                
                logger.info("App push notification sent successfully to user: {}, messageId: {}", userId, response.getMessageId());
                return true;
            } else {
                logger.error("Push notification failed: {}", response.getErrorMessage());
                return false;
            }
        } catch (Exception e) {
            logger.error("Failed to send push notification to user: {}, error: {}", userId, e.getMessage());
            return false;
        }
    }
    
    private PushRequest buildPushRequest(Long userId, String deviceToken, String title, String content, String extraData) {
        PushRequest request = new PushRequest();
        request.setUserId(userId);
        request.setDeviceToken(deviceToken);
        request.setTitle(title);
        request.setBody(content);
        request.setExtraData(extraData);
        request.setProvider(provider);
        request.setTimestamp(System.currentTimeMillis());
        return request;
    }
    
    private String getDeviceTokenByUserId(Long userId) {
        // In real implementation, would query user device registration
        // This is a placeholder
        return "device_token_user_" + userId;
    }
    
    private PushResponse sendPushNotification(PushRequest request) {
        // Simulated push notification sending
        // In production, would integrate with FCM, JPush, or other providers
        PushResponse response = new PushResponse();
        
        try {
            switch (provider.toLowerCase()) {
                case "fcm":
                    response = sendFcmNotification(request);
                    break;
                case "jpush":
                    response = sendJPushNotification(request);
                    break;
                default:
                    // Default to FCM
                    response = sendFcmNotification(request);
            }
        } catch (Exception e) {
            response.setSuccess(false);
            response.setErrorMessage(e.getMessage());
        }
        
        return response;
    }
    
    private PushResponse sendFcmNotification(PushRequest request) {
        // Simulated FCM API call
        // In production: use Firebase Admin SDK
        PushResponse response = new PushResponse();
        response.setSuccess(true);
        response.setMessageId("FCM_" + System.currentTimeMillis());
        response.setProvider("fcm");
        return response;
    }
    
    private PushResponse sendJPushNotification(PushRequest request) {
        // Simulated JPush API call
        // In production: use JPush REST API
        PushResponse response = new PushResponse();
        response.setSuccess(true);
        response.setMessageId("JPUSH_" + System.currentTimeMillis());
        response.setProvider("jpush");
        return response;
    }
    
    @Override
    public String getChannelCode() {
        return CHANNEL_CODE;
    }
    
    // Internal classes
    
    private static class PushRequest {
        private Long userId;
        private String deviceToken;
        private String title;
        private String body;
        private String extraData;
        private String provider;
        private long timestamp;
        
        // Getters and setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getDeviceToken() { return deviceToken; }
        public void setDeviceToken(String deviceToken) { this.deviceToken = deviceToken; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getBody() { return body; }
        public void setBody(String body) { this.body = body; }
        public String getExtraData() { return extraData; }
        public void setExtraData(String extraData) { this.extraData = extraData; }
        public String getProvider() { return provider; }
        public void setProvider(String provider) { this.provider = provider; }
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    }
    
    private static class PushResponse {
        private boolean success;
        private String messageId;
        private String provider;
        private String errorMessage;
        
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getMessageId() { return messageId; }
        public void setMessageId(String messageId) { this.messageId = messageId; }
        public String getProvider() { return provider; }
        public void setProvider(String provider) { this.provider = provider; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    }
    
    private static class PushRecord {
        private Long userId;
        private String deviceToken;
        private String title;
        private String messageId;
        private long sendTime;
        private String status;
        
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getDeviceToken() { return deviceToken; }
        public void setDeviceToken(String deviceToken) { this.deviceToken = deviceToken; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getMessageId() { return messageId; }
        public void setMessageId(String messageId) { this.messageId = messageId; }
        public long getSendTime() { return sendTime; }
        public void setSendTime(long sendTime) { this.sendTime = sendTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }
}