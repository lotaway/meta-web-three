package com.metawebthree.event.domain;

import com.metawebthree.event.BaseEvent;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.util.List;

/**
 * Event published when inventory is released (order cancelled or returned).
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InventoryReleasedEvent extends BaseEvent {

    private String reservationId;
    private String orderId;
    private String reason;
    private List<ReleasedItem> items;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ReleasedItem {
        private String productId;
        private Integer quantity;
        private String warehouseId;
    }
}