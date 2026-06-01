package com.metawebthree.notification.infrastructure.sender;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Email notification sender implementation
 * Uses JavaMail to send email notifications
 */
@Component
public class EmailNotificationSender implements NotificationSender {
    
    private static final Logger logger = LoggerFactory.getLogger(EmailNotificationSender.class);
    private static final String CHANNEL_CODE = "EMAIL";
    
    @Value("${notification.email.from: noreply@metawebthree.com}")
    private String fromEmail;
    
    @Value("${notification.email.enabled:true}")
    private boolean enabled;
    
    // Injected JavaMailSender (would be configured in Spring Boot)
    private final JavaMailSender mailSender;
    
    // Email sending records for tracking
    private final ConcurrentHashMap<Long, EmailRecord> emailRecordStore = new ConcurrentHashMap<>();
    
    public EmailNotificationSender(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }
    
    @Override
    public boolean send(Long userId, String title, String content, String extraData) {
        if (!enabled) {
            logger.warn("Email notification is disabled");
            return false;
        }
        
        try {
            // Get user email address
            String recipientEmail = getEmailByUserId(userId);
            
            if (recipientEmail == null || recipientEmail.isEmpty()) {
                logger.error("Cannot send email: no email address for user: {}", userId);
                return false;
            }
            
            // Build and send email
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(recipientEmail);
            message.setSubject(title);
            message.setText(content);
            message.setSentDate(new java.util.Date());
            
            // Send email through JavaMailSender
            mailSender.send(message);
            
            // Record for tracking
            EmailRecord record = new EmailRecord();
            record.setUserId(userId);
            record.setEmail(recipientEmail);
            record.setSubject(title);
            record.setSendTime(System.currentTimeMillis());
            record.setStatus("SENT");
            emailRecordStore.put(userId, record);
            
            logger.info("Email notification sent successfully to user: {}, email: {}", userId, recipientEmail);
            return true;
        } catch (Exception e) {
            logger.error("Failed to send email notification to user: {}, error: {}", userId, e.getMessage());
            return false;
        }
    }
    
    private String getEmailByUserId(Long userId) {
        // In real implementation, would query user service for email
        // This is a placeholder
        return "user" + userId + "@metawebthree.com";
    }
    
    @Override
    public String getChannelCode() {
        return CHANNEL_CODE;
    }
    
    // Internal record class
    private static class EmailRecord {
        private Long userId;
        private String email;
        private String subject;
        private long sendTime;
        private String status;
        
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getSubject() { return subject; }
        public void setSubject(String subject) { this.subject = subject; }
        public long getSendTime() { return sendTime; }
        public void setSendTime(long sendTime) { this.sendTime = sendTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }
}