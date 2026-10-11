package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("replenishment_recommendation")
public class ReplenishmentRecommendationDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private Integer currentStock;
    private Integer safetyStock;
    private Integer leadTimeDays;
    private Integer averageDailySales;
    private Integer recommendedQuantity;
    private String recommendationType;
    private String status;
    private LocalDateTime generatedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}