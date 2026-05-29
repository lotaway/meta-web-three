package com.metawebthree.invoice.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("invoice")
public class InvoiceDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String invoiceNo;
    private String orderNo;
    private Long customerId;
    private String customerName;
    private String customerTaxNo;
    private String customerAddress;
    private String customerBank;
    private String customerAccount;
    private String type;
    private String status;
    private BigDecimal amount;
    private BigDecimal taxAmount;
    private BigDecimal totalAmount;
    private String taxRate;
    private LocalDateTime issueDate;
    private String issuer;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}