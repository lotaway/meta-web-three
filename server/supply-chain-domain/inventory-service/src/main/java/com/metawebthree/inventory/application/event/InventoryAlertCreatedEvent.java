package com.metawebthree.inventory.application.event;

import com.metawebthree.inventory.application.dto.AlertNotificationDTO;
import lombok.Getter;
import org.springframework.context.ApplicationEvent;

@Getter
public class InventoryAlertCreatedEvent extends ApplicationEvent {

    private final AlertNotificationDTO notification;

    public InventoryAlertCreatedEvent(Object source, AlertNotificationDTO notification) {
        super(source);
        this.notification = notification;
    }
}