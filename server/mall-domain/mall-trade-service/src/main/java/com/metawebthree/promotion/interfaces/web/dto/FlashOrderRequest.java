package com.metawebthree.promotion.interfaces.web.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class FlashOrderRequest {
    private Long sessionId;
    private Long productId;
    private Long skuId;
    private String productName;
    private String productPic;
    private Integer quantity;
    private BigDecimal flashPrice;
    private Long memberReceiveAddressId;
    private String orderRemark;
}
