package com.metawebthree.procurement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;

@Data
@TableName("procurement_return_order_item")
public class ProcurementReturnOrderItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long returnOrderId;
    private String returnNo;
    private String sourceOrderNo;
    private String sourceOrderItemId;
    private String skuCode;
    private String productName;
    private Integer returnQuantity;
    private BigDecimal unitPrice;
    private BigDecimal totalAmount;
    private String reason;
    private String status;
}