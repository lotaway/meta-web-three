package com.metawebthree.inventory.domain.event;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class InventoryChangedEvent {
    private String eventId;
    private String skuCode;
    private Long warehouseId;
    private String changeType;
    private Integer quantity;
    private Integer beforeQuantity;
    private Integer afterQuantity;
    private String bizId;
    private String bizType;
    private LocalDateTime occurredAt;

    public static InventoryChangedEvent of(String skuCode, Long warehouseId,
            String changeType, Integer quantity, Integer before, Integer after,
            String bizId, String bizType) {
        InventoryChangedEvent event = new InventoryChangedEvent();
        event.setEventId(java.util.UUID.randomUUID().toString());
        event.setSkuCode(skuCode);
        event.setWarehouseId(warehouseId);
        event.setChangeType(changeType);
        event.setQuantity(quantity);
        event.setBeforeQuantity(before);
        event.setAfterQuantity(after);
        event.setBizId(bizId);
        event.setBizType(bizType);
        event.setOccurredAt(LocalDateTime.now());
        return event;
    }
}