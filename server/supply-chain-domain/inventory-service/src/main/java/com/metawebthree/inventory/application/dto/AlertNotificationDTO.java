package com.metawebthree.inventory.application.dto;

import lombok.Data;

@Data
public class AlertNotificationDTO {
    private String alertCode;
    private String title;
    private String description;
    private String level;
    private String skuCode;
    private String warehouseCode;
    private Integer currentQuantity;
    private Integer thresholdValue;
    private String notificationChannels;
    private String notifyUsers;
}