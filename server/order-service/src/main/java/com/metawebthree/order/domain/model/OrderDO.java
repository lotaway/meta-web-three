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
    private String orderStatus;
    private String orderType;
    private BigDecimal orderAmount;
    private String orderRemark;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}
