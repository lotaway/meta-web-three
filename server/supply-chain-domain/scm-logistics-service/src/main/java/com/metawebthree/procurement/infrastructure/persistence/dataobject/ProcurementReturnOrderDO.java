package com.metawebthree.procurement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("procurement_return_order")
public class ProcurementReturnOrderDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    private String returnNo;
    private String sourceOrderNo;
    private String sourceOrderType;
    private String supplierCode;
    private String supplierName;
    private Long warehouseId;
    private String warehouseName;
    private String returnType;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private String reason;
    private String remark;
    private String approver;
    private String approvalComment;
    private LocalDateTime approvedAt;
    private LocalDateTime expectedReturnDate;
    private LocalDateTime actualReturnDate;
    private String logisticsCompany;
    private String trackingNumber;
    private LocalDateTime shippedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}