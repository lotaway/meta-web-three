package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("inbound_order_item")
public class InboundOrderItemDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long orderId;
    private String skuCode;
    private String productName;
    private Integer planQuantity;
    private Integer actualQuantity;
    private Long locationId;
    private String status;
    private java.math.BigDecimal unitCost;
    private String batchNo;
    private LocalDateTime productionDate;
    private LocalDateTime expiryDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}