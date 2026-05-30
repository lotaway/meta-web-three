package com.metawebthree.supplier.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("supplier_reconciliation_item")
public class SupplierReconciliationItemDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long reconciliationId;
    private String orderNo;
    private LocalDate orderDate;
    private LocalDate shippedDate;
    private BigDecimal invoicedAmount;
    private BigDecimal settledAmount;
    private BigDecimal pendingAmount;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}