package com.metawebthree.inventory.application.command;

import lombok.Data;

@Data
public class CreateAlertCommand {
    private String warehouseCode;
    private String skuCode;
    private Integer thresholdValue;
    private String notificationChannels;
    private String notifyUsers;
}