package com.metawebthree.aftersale.application.dto;

import lombok.Data;

@Data
public class AfterSaleApplyDTO {
    private Long orderId;
    private Long productId;
    private Long skuId;
    private Integer quantity;
    private Integer refundAmount;
    private Integer afterSaleType;
    private String applyReason;
    private String remark;
}