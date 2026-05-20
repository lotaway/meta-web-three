package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class Location {
    private Long id;
    private Long warehouseId;
    private String locationCode;
    private String zoneCode;
    private String shelfCode;
    private Integer row;
    private Integer column;
    private Integer layer;
    private String locationType;
    private String status;
    private Integer maxWeight;
    private Integer maxVolume;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public boolean isAvailable() {
        return "IDLE".equals(status);
    }

    public void occupy() {
        this.status = "OCCUPIED";
    }

    public void release() {
        this.status = "IDLE";
    }
}