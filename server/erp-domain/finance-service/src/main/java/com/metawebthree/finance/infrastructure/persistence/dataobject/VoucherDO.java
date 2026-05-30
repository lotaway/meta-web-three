package com.metawebthree.finance.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("finance_voucher")
public class VoucherDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String voucherNo;
    private String type;
    private LocalDateTime voucherDate;
    private String description;
    private String status;
    private String currency;
    private BigDecimal exchangeRate;
    private String createdBy;
    private String approvedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}