package com.metawebthree.order.DO;

import java.math.BigDecimal;

import com.baomidou.mybatisplus.annotation.TableName;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
@TableName("order")
public class OrderDO {
    private Long id;
    private Long userId;
    private String orderNo;
    private String orderStatus;
    private String orderType;
    private BigDecimal orderAmount;
    private String orderRemark;
}
