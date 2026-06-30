package com.metawebthree.finance.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;

@Data
@TableName("finance_voucher_line")
public class VoucherLineDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long voucherId;
    private Long subjectId;
    private BigDecimal debitAmount;
    private BigDecimal creditAmount;
    private String foreignCurrency;
    private BigDecimal foreignDebitAmount;
    private BigDecimal foreignCreditAmount;
}