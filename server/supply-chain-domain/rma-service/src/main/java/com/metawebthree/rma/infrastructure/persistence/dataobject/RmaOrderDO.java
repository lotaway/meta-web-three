package com.metawebthree.rma.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("rma_order")
public class RmaOrderDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String rmaNo;
    private String orderNo;
    private String returnType;
    private String status;
    private Long customerId;
    private String customerName;
    private String contactPhone;
    private String reasonCode;
    private String reasonDescription;
    private Long warehouseId;
    private Integer totalQuantity;
    private BigDecimal totalAmount;
    private String currency;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}
