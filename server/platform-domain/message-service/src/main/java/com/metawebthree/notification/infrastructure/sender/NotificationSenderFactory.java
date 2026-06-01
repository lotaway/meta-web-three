package com.metawebthree.notification.infrastructure.sender;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Notification sender factory
 * Manages and routes notifications to appropriate sender based on channel
 */
@Component
public class NotificationSenderFactory {
    
    private static final Logger logger = LoggerFactory.getLogger(NotificationSenderFactory.class);
    
    private final Map<String, NotificationSender> senderMap = new HashMap<>();
    
    public NotificationSenderFactory(List<NotificationSender> senders) {
        // Register all available senders
        for (NotificationSender sender : senders) {
            senderMap.put(sender.getChannelCode(), sender);
            logger.info("Registered notification sender for channel: {}", sender.getChannelCode());
        }
    }
    
    /**
     * Get sender by channel code
     * @param channelCode Channel code (IN_SITE, SMS, EMAIL, APP)
     * @return Notification sender or null if not found
     */
    public NotificationSender getSender(String channelCode) {
        NotificationSender sender = senderMap.get(channelCode);
        if (sender == null) {
            logger.warn("No sender found for channel: {}", channelCode);
        }
        return sender;
    }
    
    /**
     * Check if sender exists for given channel
     * @param channelCode Channel code
     * @return true if sender exists
     */
    public boolean hasSender(String channelCode) {
        return senderMap.containsKey(channelCode);
    }
    
    /**
     * Get all available channel codes
     * @return List of channel codes
     */
    public Map<String, NotificationSender> getAllSenders() {
        return new HashMap<>(senderMap);
    }
}