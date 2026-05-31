package com.metawebthree.inventory.application;

import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertRepository;
import com.metawebthree.inventory.application.command.CreateAlertCommand;
import com.metawebthree.inventory.application.dto.AlertNotificationDTO;
import com.metawebthree.inventory.application.event.InventoryAlertCreatedEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class InventoryAlertAppService {

    private final InventoryAlertRepository alertRepository;
    private final ApplicationEventPublisher eventPublisher;

    @Transactional
    public InventoryAlert createAlert(InventoryAlert alert, InventoryAlertConfig config) {
        // 保存预警记录
        InventoryAlert savedAlert = alertRepository.save(alert);
        
        log.info("创建库存预警成功: alertCode={}, skuCode={}, level={}", 
                alert.getAlertCode(), alert.getSkuCode(), alert.getLevel());
        
        // 发布预警创建事件，用于发送通知
        if (config != null && config.getNotificationChannels() != null) {
            AlertNotificationDTO notification = buildNotification(savedAlert, config);
            InventoryAlertCreatedEvent event = new InventoryAlertCreatedEvent(this, notification);
            eventPublisher.publishEvent(event);
        }
        
        return savedAlert;
    }

    @Transactional
    public void acknowledgeAlert(Long alertId, String userId) {
        InventoryAlert alert = alertRepository.findById(alertId);
        if (alert == null) {
            throw new IllegalArgumentException("预警记录不存在: " + alertId);
        }
        alert.acknowledge(userId);
        alertRepository.save(alert);
        
        log.info("预警已确认: alertCode={}, acknowledgedBy={}", alert.getAlertCode(), userId);
    }

    @Transactional
    public void resolveAlert(Long alertId, String userId, String solution) {
        InventoryAlert alert = alertRepository.findById(alertId);
        if (alert == null) {
            throw new IllegalArgumentException("预警记录不存在: " + alertId);
        }
        alert.resolve(userId, solution);
        alertRepository.save(alert);
        
        log.info("预警已解决: alertCode={}, resolvedBy={}", alert.getAlertCode(), userId);
    }

    public List<InventoryAlert> getActiveAlerts() {
        return alertRepository.findActiveAlerts();
    }

    public List<InventoryAlert> getAlertsBySkuCode(String skuCode) {
        return alertRepository.findBySkuCodeAndStatus(skuCode, null);
    }

    private AlertNotificationDTO buildNotification(InventoryAlert alert, InventoryAlertConfig config) {
        AlertNotificationDTO notification = new AlertNotificationDTO();
        notification.setAlertCode(alert.getAlertCode());
        notification.setTitle(alert.getTitle());
        notification.setDescription(alert.getDescription());
        notification.setLevel(alert.getLevel().name());
        notification.setSkuCode(alert.getSkuCode());
        notification.setWarehouseCode(alert.getWarehouseCode());
        notification.setCurrentQuantity(alert.getCurrentQuantity());
        notification.setThresholdValue(alert.getThresholdValue());
        notification.setNotificationChannels(config.getNotificationChannels());
        notification.setNotifyUsers(config.getNotifyUsers());
        return notification;
    }
}