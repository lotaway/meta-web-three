package com.metawebthree.groupbuying.domain.model;

import java.math.BigDecimal;
import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.IdType;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("tb_group_buy_order")
public class GroupBuyOrderDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long teamId;
    private Long activityId;
    private Long userId;
    private String orderNo;
    private Long orderId;
    private Long productId;
    private Integer quantity;
    private BigDecimal unitPrice;
    private BigDecimal totalAmount;
    private String status;
    private Boolean isLeader;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}