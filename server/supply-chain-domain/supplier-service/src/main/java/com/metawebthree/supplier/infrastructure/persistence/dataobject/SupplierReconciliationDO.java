package com.metawebthree.supplier.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("supplier_reconciliation")
public class SupplierReconciliationDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String reconciliationNo;
    private String supplierCode;
    private LocalDate periodStart;
    private LocalDate periodEnd;
    private Integer orderCount;
    private BigDecimal totalAmount;
    private BigDecimal shippedAmount;
    private BigDecimal invoicedAmount;
    private BigDecimal settledAmount;
    private BigDecimal pendingAmount;
    private String currency;
    private String status;
    private LocalDateTime submittedAt;
    private LocalDateTime confirmedAt;
    private String confirmedBy;
    private LocalDateTime paidAt;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}