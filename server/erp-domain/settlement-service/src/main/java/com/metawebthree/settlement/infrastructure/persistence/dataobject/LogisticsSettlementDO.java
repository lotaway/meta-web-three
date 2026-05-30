package com.metawebthree.settlement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("logistics_settlement")
public class LogisticsSettlementDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String settlementNo;
    private String trackingNo;
    private String orderNo;
    private Long carrierId;
    private String carrierName;
    private BigDecimal freight;
    private BigDecimal handlingFee;
    private BigDecimal discount;
    private BigDecimal totalAmount;
    private String status;
    private String billingCycle;
    private LocalDateTime settlementDate;
    private LocalDateTime paidAt;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    @TableLogic
    private Integer deleted;
}