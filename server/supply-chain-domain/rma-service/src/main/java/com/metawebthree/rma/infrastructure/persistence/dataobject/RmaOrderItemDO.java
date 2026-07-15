package com.metawebthree.rma.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("rma_order_item")
public class RmaOrderItemDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long rmaId;
    private String skuCode;
    private String skuName;
    private Integer expectedQuantity;
    private Integer inspectedQuantity;
    private Integer acceptedQuantity;
    private BigDecimal unitPrice;
    private String reasonCode;
    private String reasonDescription;
    private LocalDateTime createdAt;
}
