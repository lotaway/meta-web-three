package com.metawebthree.inventoryalert.application.dto;

import lombok.Data;

@Data
public class ThresholdConfigDTO {
    private Long productId;
    private Long skuId;
    private Integer minThreshold;
    private Integer maxThreshold;
    private Integer alertLevel;
    private Boolean enabled;
}