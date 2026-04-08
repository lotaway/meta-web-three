package com.metawebthree.order.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDateTime;

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
@TableName("tb_order")
@Schema(description = "订单")
public class OrderDO {
    @TableId(type = IdType.INPUT)
    @Schema(description = "订单ID")
    private Long id;
    @Schema(description = "用户ID")
    private Long userId;
    @Schema(description = "订单编号")
    private String orderNo;
    @Schema(description = "订单状态")
    private String orderStatus;
    @Schema(description = "订单类型")
    private String orderType;
    @Schema(description = "支付方式")
    private Integer paymentType;
    @Schema(description = "支付时间")
    private LocalDateTime paymentTime;
    @Schema(description = "删除状态")
    private Integer deleteStatus;
    @Schema(description = "订单金额")
    private BigDecimal orderAmount;
    @Schema(description = "订单备注")
    private String orderRemark;
    @Schema(description = "创建时间")
    private Timestamp createdAt;
    @Schema(description = "更新时间")
    private Timestamp updatedAt;
}
