package com.metawebthree.order.domain.model;

import java.math.BigDecimal;
import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
@TableName("tb_order_item")
public class OrderItemDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long orderId;
    private Long productId;
    private String productName;
    private Long skuId;
    private Integer quantity;
    private BigDecimal unitPrice;
    private BigDecimal totalPrice;
    private String imageUrl;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}

