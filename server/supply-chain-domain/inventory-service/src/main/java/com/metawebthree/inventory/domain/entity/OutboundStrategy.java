package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class OutboundStrategy {
    private Long id;
    private String strategyCode;
    private String strategyName;
    private String strategyType;
    private Long warehouseId;
    private String warehouseCode;
    private String skuCode;
    private String skuCodePattern;
    private Integer priority;
    private String specificBatchNo;
    private Boolean isActive;
    private String remark;
    private String creator;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;

    public boolean matchesWarehouse(Long warehouseId) {
        if (this.warehouseId == null) {
            return true;
        }
        return this.warehouseId.equals(warehouseId);
    }

    public boolean matchesSku(String skuCode) {
        if (skuCode == null) {
            return false;
        }
        if (this.skuCode != null && this.skuCode.equals(skuCode)) {
            return true;
        }
        if (this.skuCodePattern != null && skuCode.matches(this.skuCodePattern)) {
            return true;
        }
        return this.skuCode == null && this.skuCodePattern == null;
    }

    public boolean isFifo() {
        return "FIFO".equals(strategyType);
    }

    public boolean isLifo() {
        return "LIFO".equals(strategyType);
    }

    public boolean isSpecificBatch() {
        return "SPECIFIC_BATCH".equals(strategyType);
    }
}