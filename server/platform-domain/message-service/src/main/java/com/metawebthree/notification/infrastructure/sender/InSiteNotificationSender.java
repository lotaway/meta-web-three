package com.metawebthree.notification.infrastructure.sender;

import com.metawebthree.notification.domain.model.NotificationDO;
import com.metawebthree.notification.domain.repository.NotificationRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * In-site notification sender implementation
 * Stores notification in the platform's message system
 */
@Component
public class InSiteNotificationSender implements NotificationSender {
    
    private static final Logger logger = LoggerFactory.getLogger(InSiteNotificationSender.class);
    private static final String CHANNEL_CODE = "IN_SITE";
    
    private final NotificationRepository notificationRepository;
    
    public InSiteNotificationSender(NotificationRepository notificationRepository) {
        this.notificationRepository = notificationRepository;
    }
    
    @Override
    public boolean send(Long userId, String title, String content, String extraData) {
        try {
            // Create in-site notification record
            NotificationDO notification = new NotificationDO();
            notification.setUserId(userId);
            notification.setTitle(title);
            notification.setContent(content);
            notification.setChannel(CHANNEL_CODE);
            notification.setNotificationType("IN_SITE");
            notification.setExtraData(extraData);
            notification.setSendTime(LocalDateTime.now());
            notification.setCreateTime(LocalDateTime.now());
            notification.setUpdateTime(LocalDateTime.now());
            
            notificationRepository.save(notification);
            
            logger.info("In-site notification sent successfully to user: {}", userId);
            return true;
        } catch (Exception e) {
            logger.error("Failed to send in-site notification to user: {}, error: {}", userId, e.getMessage());
            return false;
        }
    }
    
    @Override
    public String getChannelCode() {
        return CHANNEL_CODE;
    }
}