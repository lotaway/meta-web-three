package com.metawebthree.settlement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("settlement_order")
public class SettlementOrderDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String settlementNo;
    private String orderNo;
    private Long merchantId;
    private String merchantName;
    private BigDecimal orderAmount;
    private BigDecimal settlementAmount;
    private BigDecimal commissionAmount;
    private BigDecimal refundAmount;
    private String status;
    private String channel;
    private LocalDateTime settlementDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private String description;
    private Integer version;
}