package com.metawebthree.dom.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("dom_order")
public class DomOrderDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String domOrderNo;
    private String originalOrderNo;
    private String customerId;
    private String customerName;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private Integer priority;
    private String sourcingStrategy;
    private String region;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}
