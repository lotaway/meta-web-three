package com.metaweb.datasource.pipeline.model;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class InventoryEvent {
    private String eventId;
    private String eventType; // STOCK_IN, STOCK_OUT, ADJUST, ALERT
    private Long productId;
    private String productName;
    private Integer quantity;
    private Integer availableQty;
    private Integer reservedQty;
    private String warehouseId;
    private LocalDateTime eventTime;
    private String operator;
    private String remark;
    
    public InventoryEvent() {
    }
    
    public InventoryEvent(String eventId, String eventType, Long productId) {
        this.eventId = eventId;
        this.eventType = eventType;
        this.productId = productId;
        this.eventTime = LocalDateTime.now();
    }
}
