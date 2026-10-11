package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("stocktake_order_item")
public class StocktakeOrderItemDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long stocktakeOrderId;
    private String skuCode;
    private String skuName;
    private String unit;
    private BigDecimal systemQuantity;
    private BigDecimal countedQuantity;
    private BigDecimal discrepancyQuantity;
    private BigDecimal discrepancyAmount;
    private String discrepancyReason;
    private String status;
    private String counter;
    private LocalDateTime countedAt;
    private String checker;
    private LocalDateTime checkedAt;
    private String adjuster;
    private LocalDateTime adjustedAt;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}
