package com.metawebthree.inventory.application.listener;

import com.metawebthree.inventory.application.dto.AlertNotificationDTO;
import com.metawebthree.inventory.application.event.InventoryAlertCreatedEvent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Component
@Slf4j
public class InventoryAlertNotificationListener {

    private final RestTemplate restTemplate;
    
    @Value("${notification.email-service-url:}")
    private String emailServiceUrl;
    
    @Value("${notification.sms-service-url:}")
    private String smsServiceUrl;
    
    @Value("${notification.message-service-url:}")
    private String messageServiceUrl;
    
    @Value("${notification.dingtalk-webhook-url:}")
    private String dingTalkWebhookUrl;

    public InventoryAlertNotificationListener(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Async
    @EventListener
    public void handleAlertCreated(InventoryAlertCreatedEvent event) {
        AlertNotificationDTO notification = event.getNotification();
        
        log.info("Received inventory alert created event: alertCode={}, channels={}", 
                notification.getAlertCode(), notification.getNotificationChannels());
        
        // Send notification based on channel
        String[] channels = notification.getNotificationChannels().split(",");
        
        for (String channel : channels) {
            try {
                sendNotification(channel.trim(), notification);
            } catch (Exception e) {
                log.error("Failed to send alert notification: channel={}, error={}", channel, e.getMessage(), e);
            }
        }
    }

    private void sendNotification(String channel, AlertNotificationDTO notification) {
        switch (channel.toUpperCase()) {
            case "EMAIL":
                sendEmailNotification(notification);
                break;
            case "SMS":
                sendSmsNotification(notification);
                break;
            case "IN_APP":
                sendInAppNotification(notification);
                break;
            case "DINGTALK":
                sendDingTalkNotification(notification);
                break;
            default:
                log.warn("Unknown notification channel: {}", channel);
        }
    }

    private void sendEmailNotification(AlertNotificationDTO notification) {
        if (emailServiceUrl == null || emailServiceUrl.isEmpty()) {
            log.warn("Email service URL not configured, cannot send email notification");
            return;
        }
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("to", notification.getNotifyUsers());
            payload.put("subject", notification.getTitle());
            payload.put("body", notification.getDescription());
            restTemplate.postForEntity(emailServiceUrl + "/send", payload, Void.class);
            log.info("Email notification sent successfully: alertCode={}, recipient={}", 
                    notification.getAlertCode(), notification.getNotifyUsers());
        } catch (Exception e) {
            log.error("Failed to send email notification: alertCode={}, error={}", 
                    notification.getAlertCode(), e.getMessage());
        }
    }

    private void sendSmsNotification(AlertNotificationDTO notification) {
        if (smsServiceUrl == null || smsServiceUrl.isEmpty()) {
            log.warn("SMS service URL not configured, cannot send SMS notification");
            return;
        }
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("phone", notification.getNotifyUsers());
            payload.put("content", notification.getDescription());
            restTemplate.postForEntity(smsServiceUrl + "/send", payload, Void.class);
            log.info("SMS notification sent successfully: alertCode={}, recipient={}", 
                    notification.getAlertCode(), notification.getNotifyUsers());
        } catch (Exception e) {
            log.error("Failed to send SMS notification: alertCode={}, error={}", 
                    notification.getAlertCode(), e.getMessage());
        }
    }

    private void sendInAppNotification(AlertNotificationDTO notification) {
        if (messageServiceUrl == null || messageServiceUrl.isEmpty()) {
            log.warn("Message service URL not configured, cannot send in-app notification");
            return;
        }
        try {
            String url = messageServiceUrl + "/notification/send";
            Map<String, Object> payload = new HashMap<>();
            payload.put("title", notification.getTitle());
            payload.put("content", notification.getDescription());
            payload.put("type", "INVENTORY_ALERT");
            payload.put("relatedId", notification.getAlertCode());
            payload.put("icon", "warning");
            restTemplate.postForEntity(url, payload, Void.class);
            log.info("In-app notification sent successfully: alertCode={}", notification.getAlertCode());
        } catch (Exception e) {
            log.error("Failed to send in-app notification: alertCode={}, error={}", 
                    notification.getAlertCode(), e.getMessage());
        }
    }

    private void sendDingTalkNotification(AlertNotificationDTO notification) {
        if (dingTalkWebhookUrl == null || dingTalkWebhookUrl.isEmpty()) {
            return;
        }
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("msgtype", "text");
            Map<String, Object> text = new HashMap<>();
            text.put("content", String.format("【库存预警】%s\n%s", 
                    notification.getTitle(), notification.getDescription()));
            payload.put("text", text);
            restTemplate.postForEntity(dingTalkWebhookUrl, payload, Void.class);
            log.info("DingTalk notification sent successfully: alertCode={}", notification.getAlertCode());
        } catch (Exception e) {
            log.error("Failed to send DingTalk notification: alertCode={}, error={}", 
                    notification.getAlertCode(), e.getMessage());
        }
    }
}