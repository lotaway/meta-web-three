package com.metawebthree.procurement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("procurement_order")
public class ProcurementOrderDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String orderNo;
    private String supplierCode;
    private Long warehouseId;
    private String purchaseType; // STOCK/PRODUCTION
    private String status; // DRAFT/PENDING/APPROVED/REJECTED/COMPLETED/CANCELLED
    private BigDecimal totalAmount;
    private String currency;
    private String paymentTerms;
    private String deliveryTerms;
    private String remark;
    private String approver;
    private LocalDateTime approvedAt;
    private LocalDateTime expectedDeliveryDate;
    private LocalDateTime actualDeliveryDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}