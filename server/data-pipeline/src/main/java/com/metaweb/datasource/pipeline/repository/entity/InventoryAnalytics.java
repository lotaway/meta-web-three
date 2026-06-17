package com.metaweb.datasource.pipeline.repository.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("inventory_analytics")
public class InventoryAnalytics {
    private String eventId;
    private String eventType;
    private Long productId;
    private String productName;
    private Integer quantity;
    private Integer availableQty;
    private Integer reservedQty;
    private String warehouseId;
    private LocalDateTime eventTime;
    private String operator;
    private String remark;
    private LocalDateTime processedTime;
}
