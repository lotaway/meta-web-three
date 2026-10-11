package com.metawebthree.dom.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("dom_order_line")
public class DomOrderLineDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long domOrderId;
    private String skuCode;
    private String skuName;
    private Integer quantity;
    private Integer fulfilledQuantity;
    private Long warehouseId;
    private String warehouseName;
    private BigDecimal unitPrice;
    private String status;
    private LocalDateTime createdAt;
}
