package com.metawebthree.warehouse.application.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class WarehouseDTO {
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
    private Integer availableCapacity;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}