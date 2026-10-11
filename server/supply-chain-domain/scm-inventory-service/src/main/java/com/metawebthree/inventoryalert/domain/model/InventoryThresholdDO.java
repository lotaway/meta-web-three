package com.metawebthree.inventoryalert.domain.model;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class InventoryThresholdDO {
    private Long id;
    private Long productId;
    private Long skuId;
    private Integer minThreshold;
    private Integer maxThreshold;
    private Integer alertLevel;
    private Boolean enabled;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}