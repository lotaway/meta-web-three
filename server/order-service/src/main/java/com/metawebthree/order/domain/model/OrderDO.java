package com.metawebthree.order.domain.model;

import java.math.BigDecimal;
import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.IdType;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
@TableName("tb_order")
public class OrderDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long userId;
    private String orderNo;
    private OrderStatus status;
    private OrderType type;
    private BigDecimal orderAmount;
    private String orderRemark;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}

enum OrderStatus {
    PENDING, CONFIRMED, PAID, SHIPPED, COMPLETED, CANCELLED, REFUNDED
}

enum OrderType {
    NORMAL, GROUP_BUY, FLASH_SALE, PRE_ORDER
}
