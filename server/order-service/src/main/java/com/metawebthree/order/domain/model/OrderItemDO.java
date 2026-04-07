package com.metawebthree.order.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
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
@Schema(description = "订单商品")
public class OrderItemDO {
    @TableId(type = IdType.INPUT)
    @Schema(description = "商品ID")
    private Long id;
    @Schema(description = "订单ID")
    private Long orderId;
    @Schema(description = "商品ID")
    private Long productId;
    @Schema(description = "商品名称")
    private String productName;
    @Schema(description = "SKU ID")
    private Long skuId;
    @Schema(description = "数量")
    private Integer quantity;
    @Schema(description = "单价")
    private BigDecimal unitPrice;
    @Schema(description = "总价")
    private BigDecimal totalPrice;
    @Schema(description = "图片URL")
    private String imageUrl;
    @Schema(description = "创建时间")
    private Timestamp createdAt;
    @Schema(description = "更新时间")
    private Timestamp updatedAt;
}

