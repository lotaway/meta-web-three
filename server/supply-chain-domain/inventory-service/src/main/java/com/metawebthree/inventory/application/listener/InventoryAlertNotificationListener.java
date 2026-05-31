package com.metawebthree.inventory.application.listener;

import com.metawebthree.inventory.application.dto.AlertNotificationDTO;
import com.metawebthree.inventory.application.event.InventoryAlertCreatedEvent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class InventoryAlertNotificationListener {

    @Async
    @EventListener
    public void handleAlertCreated(InventoryAlertCreatedEvent event) {
        AlertNotificationDTO notification = event.getNotification();
        
        log.info("收到库存预警创建事件: alertCode={}, channels={}", 
                notification.getAlertCode(), notification.getNotificationChannels());
        
        // 根据通知渠道发送通知
        String[] channels = notification.getNotificationChannels().split(",");
        
        for (String channel : channels) {
            try {
                sendNotification(channel.trim(), notification);
            } catch (Exception e) {
                log.error("发送预警通知失败: channel={}, error={}", channel, e.getMessage(), e);
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
                log.warn("未知的通知渠道: {}", channel);
        }
    }

    private void sendEmailNotification(AlertNotificationDTO notification) {
        // 发送邮件通知（待集成邮件服务后替换为真实调用）
        // 示例: emailService.send(notification.getRecipient(), notification.getTitle(), notification.getContent());
        log.info("发送邮件通知: alertCode={}, title={}", 
                notification.getAlertCode(), notification.getTitle());
    }

    private void sendSmsNotification(AlertNotificationDTO notification) {
        // 发送短信通知（待集成短信服务后替换为真实调用）
        // 示例: smsService.send(notification.getRecipient(), notification.getContent());
        log.info("发送短信通知: alertCode={}, title={}", 
                notification.getAlertCode(), notification.getTitle());
    }

    private void sendInAppNotification(AlertNotificationDTO notification) {
        // 发送站内信通知（待集成站内信服务后替换为真实调用）
        // 示例: messageService.sendInApp(notification.getRecipient(), notification.getTitle(), notification