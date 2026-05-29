package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("inventory")
public class InventoryDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private Integer totalQuantity;
    private Integer availableQuantity;
    private Integer reservedQuantity;
    private Integer defectiveQuantity;
    private BigDecimal unitCost;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}