package com.metawebthree.settlement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("split_rule")
public class SplitRuleDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String ruleNo;
    private String ruleName;
    private String type;
    private Long merchantId;
    private BigDecimal ratio;
    private BigDecimal fixedAmount;
    private BigDecimal minAmount;
    private BigDecimal maxAmount;
    private String status;
    private Integer priority;
    private LocalDateTime effectiveDate;
    private LocalDateTime expireDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}