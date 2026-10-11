package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("inventory_batch")
public class InventoryBatchDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private String batchNo;
    private Integer quantity;
    private Integer availableQuantity;
    private Integer reservedQuantity;
    private Integer pickedQuantity;
    private LocalDateTime inboundDate;
    private LocalDateTime productionDate;
    private LocalDateTime expiryDate;
    private BigDecimal unitCost;
    private String locationCode;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}