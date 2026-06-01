package com.metawebthree.notification.infrastructure.sender;

/**
 * Notification sender interface
 * Defines the contract for sending notifications through different channels
 */
public interface NotificationSender {
    
    /**
     * Send notification through specific channel
     * @param userId User ID
     * @param title Notification title
     * @param content Notification content
     * @param extraData Additional data
     * @return true if sent successfully, false otherwise
     */
    boolean send(Long userId, String title, String content, String extraData);
    
    /**
     * Get channel code that this sender handles
     * @return channel code
     */
    String getChannelCode();
}