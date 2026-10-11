package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ReservationRecord {
    private Long id;
    private String bizId;
    private String skuCode;
    private Long warehouseId;
    private Integer quantity;
    private String status; // PENDING, CONFIRMED, CANCELLED
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}