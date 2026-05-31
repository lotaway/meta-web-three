package com.metawebthree.inventory.domain.entity.alert;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class InventoryAlertConfig {
    private Long id;
    private String configCode;
    private String warehouseCode;
    private String skuCode;
    private Integer safetyStockThreshold;
    private AlertLevel level;
    private Boolean enabled;
    private Integer cooldownMinutes;
    private String notificationChannels;
    private String notifyUsers;
    private String createdBy;
    private LocalDateTime createdAt;
    private String updatedBy;
    private LocalDateTime updatedAt;
    private Integer version;

    public enum AlertLevel {
        INFO, WARNING, ERROR, CRITICAL
    }

    public boolean isEnabled() {
        return Boolean.TRUE.equals(enabled);
    }
}