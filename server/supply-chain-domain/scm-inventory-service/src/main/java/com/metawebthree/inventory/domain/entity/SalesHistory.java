package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.time.LocalDate;

@Data
public class SalesHistory {
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private LocalDate salesDate;
    private Integer quantity;
    private String salesChannel;
    private LocalDate createdAt;

    public boolean isWithinDays(Integer days) {
        if (days == null || salesDate == null) {
            return false;
        }
        return salesDate.isAfter(LocalDate.now().minusDays(days));
    }
}