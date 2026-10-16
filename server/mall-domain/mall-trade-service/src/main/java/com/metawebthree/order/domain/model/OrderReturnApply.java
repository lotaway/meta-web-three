package com.metawebthree.order.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import java.time.LocalDateTime;

@Getter
@Builder
@Schema(description = "退货申请")
public class OrderReturnApply {
    @Schema(description = "申请ID")
    private final Long id;
    @Schema(description = "订单ID")
    private final Long orderId;
    @Schema(description = "公司地址ID")
    private final Long companyAddressId;
    @Schema(description = "商品ID")
    private final Long productId;
    @Schema(description = "订单编号")
    private final String orderNo;
    @Schema(description = "申请时间")
    private final LocalDateTime createTime;
    @Schema(description = "会员用户名")
    private final String memberUsername;
    @Schema(description = "退货金额")
    private final Double returnAmount;
    @Schema(description = "退货人姓名")
    private final String returnName;
    @Schema(description = "退货人电话")
    private final String returnPhone;
    @Schema(description = "状态")
    private final Integer status;
    @Schema(description = "处理时间")
    private final LocalDateTime handleTime;
    @Schema(description = "商品图片")
    private final String productPic;
    @Schema(description = "商品名称")
    private final String productName;
    @Schema(description = "商品品牌")
    private final String productBrand;
    @Schema(description = "商品属性")
    private final String productAttr;
    @Schema(description = "商品数量")
    private final Integer productCount;
    @Schema(description = "商品价格")
    private final Double productPrice;
    @Schema(description = "实际商品价格")
    private final Double productRealPrice;
    @Schema(description = "退货原因")
    private final String reason;
    @Schema(description = "描述")
    private final String description;
    @Schema(description = "凭证图片")
    private final String proofPics;
    @Schema(description = "处理备注")
    private final String handleNote;
    @Schema(description = "处理人")
    private final String handleMan;
    @Schema(description = "收货人")
    private final String receiveMan;
    @Schema(description = "收货时间")
    private final LocalDateTime receiveTime;
    @Schema(description = "收货备注")
    private final String receiveNote;

    public boolean isPending() {
        return status != null && status == 0;
    }

    public boolean isCompleted() {
        return status != null && status == 2;
    }
}
