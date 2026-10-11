package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class Warehouse {
    private Long id;
    private String warehouseCode;
    private String warehouseName;
    private String warehouseType;
    private String province;
    private String city;
    private String district;
    private String address;
    private String contact;
    private String phone;
    private Integer totalCapacity;
    private Integer usedCapacity;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;

    public boolean isAvailable() {
        return "ACTIVE".equals(status);
    }

    public Integer getAvailableCapacity() {
        return totalCapacity - usedCapacity;
    }

    public boolean canStore(Integer quantity) {
        return isAvailable() && getAvailableCapacity() >= quantity;
    }

    public void increaseCapacity(Integer quantity) {
        usedCapacity += quantity;
    }

    public void decreaseCapacity(Integer quantity) {
        usedCapacity = Math.max(0, usedCapacity - quantity);
    }
}