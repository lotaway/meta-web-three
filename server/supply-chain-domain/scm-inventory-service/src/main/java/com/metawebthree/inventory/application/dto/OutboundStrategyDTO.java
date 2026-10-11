package com.metawebthree.inventory.application.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class OutboundStrategyDTO {
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
}