package com.metawebthree.inventory.application.event;

import com.metawebthree.inventory.application.dto.AlertNotificationDTO;
import lombok.Getter;

@Getter
public class InventoryAlertCreatedEvent {

    private final String eventType;
    private final AlertNotificationDTO notification;

    public InventoryAlertCreatedEvent(String eventType, AlertNotificationDTO notification) {
        this.eventType = eventType;
        this.notification = notification;
    }
}