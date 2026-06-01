package com.metawebthree.aftersale.domain.model;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class AfterSaleOrderDO {
    private Long id;
    private Long orderId;
    private String orderNo;
    private Long userId;
    private Long productId;
    private Long skuId;
    private String productName;
    private String productImage;
    private Integer quantity;
    private Integer refundAmount;
    private Integer afterSaleType;
    private Integer afterSaleStatus;
    private String applyReason;
    private String rejectReason;
    private LocalDateTime applyTime;
    private LocalDateTime processTime;
    private LocalDateTime completeTime;
    private String remark;
}