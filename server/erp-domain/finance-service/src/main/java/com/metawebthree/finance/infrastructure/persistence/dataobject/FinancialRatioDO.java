package com.metawebthree.finance.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("finance_financial_ratio")
public class FinancialRatioDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String ratioType;
    private BigDecimal value;
    private String period;
    private LocalDateTime calculatedAt;
    private Integer version;
}