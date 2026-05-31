package com.metawebthree.inventory.domain.service;

import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class InventoryAlertDomainService {

    /**
     * 检查库存是否需要预警
     */
    public List<InventoryAlert> checkInventoryAlerts(Inventory inventory, InventoryAlertConfig config) {
        List<InventoryAlert> alerts = new ArrayList<>();
        
        if (config == null || !config.isEnabled()) {
            return alerts;
        }
        
        // 检查低库存预警
        if (inventory.getAvailableQuantity() < inventory.getSafetyStock()) {
            InventoryAlert alert = createLowStockAlert(inventory, config);
            alerts.add(alert);
            log.info("生成低库存预警: skuCode={}, available={}, safetyStock={}", 
                    inventory.getSkuCode(), inventory.getAvailableQuantity(), inventory.getSafetyStock());
        }
        
        return alerts;
    }

    /**
     * 根据库存数据创建低库存预警
     */
    private InventoryAlert createLowStockAlert(Inventory inventory, InventoryAlertConfig config) {
        String alertCode = generateAlertCode(inventory.getSkuCode());
        String title = String.format("库存不足预警 - %s", inventory.getSkuCode());
        String description = String.format("商品 %s 在仓库 %d 的可用库存为 %d，低于安全库存 %d", 
                inventory.getSkuCode(), 
                inventory.getWarehouseId(),
                inventory.getAvailableQuantity(), 
                inventory.getSafetyStock());
        
        InventoryAlert.AlertLevel level = calculateAlertLevel(
                inventory.getAvailableQuantity(), 
                inventory.getSafetyStock());
        
        InventoryAlert alert = new InventoryAlert(
                alertCode,
                inventory.getWarehouseId().toString(),
                inventory.getSkuCode(),
                InventoryAlert.AlertType.LOW_STOCK,
                level,
                title,
                inventory.getAvailableQuantity(),
                inventory.getSafetyStock()
        );
        alert.setDescription(description);
        
        return alert;
    }

    /**
     * 计算预警级别
     */
    private InventoryAlert.AlertLevel calculateAlertLevel(Integer availableQuantity, Integer safetyStock) {
        if (availableQuantity == null || safetyStock == null || safetyStock == 0) {
            return InventoryAlert.AlertLevel.WARNING;
        }
        
        double ratio = (double) availableQuantity / safetyStock;
        
        if (ratio <= 0.25) {
            return InventoryAlert.AlertLevel.CRITICAL;
        } else if (ratio <= 0.5) {
            return InventoryAlert.AlertLevel.ERROR;
        } else if (ratio <= 0.75) {
            return InventoryAlert.AlertLevel.WARNING;
        } else {
            return InventoryAlert.AlertLevel.INFO;
        }
    }

    /**
     * 生成预警编号
     */
    private String generateAlertCode(String skuCode) {
        return "INV-ALERT-" + skuCode + "-" + System.currentTimeMillis();
    }

    /**
     * 检查是否在冷却期内（避免重复预警）
     */
    public boolean isInCooldown(InventoryAlert lastAlert, Integer cooldownMinutes) {
        if (lastAlert == null || cooldownMinutes == null) {
            return false;
        }
        
        long minutesSinceLastAlert = java.time.Duration.between(
                lastAlert.getCreatedAt(), 
                java.time.LocalDateTime.now()
        ).toMinutes();
        
        return minutesSinceLastAlert < cooldownMinutes;
    }
}