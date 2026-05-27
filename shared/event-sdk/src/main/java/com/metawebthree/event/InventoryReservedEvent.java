package com.metawebthree.event;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.time.Instant;
import java.util.List;

/**
 * Event published when inventory is reserved for an order.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InventoryReservedEvent extends BaseEvent {

    private String reservationId;
    private String orderId;
    private List<ReservedItem> items;
    private Instant expiresAt;

    public static Builder builder() { return Builder.builder(); }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ReservedItem {
        private String productId;
        private Integer quantity;
        private String warehouseId;
    }
}