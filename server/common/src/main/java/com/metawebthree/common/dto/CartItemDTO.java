package com.metawebthree.common.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.math.BigDecimal;
import java.util.Date;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CartItemDTO {
    private Long id;
    private Long productId;
    private Long productSkuId;
    private Long memberId;
    private Integer quantity;
    private BigDecimal price;
    private String productPic;
    private String productName;
    private String productSubTitle;
    private String productSkuCode;
    private String memberNickname;
    private Date createDate;
    private Date modifyDate;
    private Integer deleteStatus;
    private Long productCategoryId;
    private String productBrand;
    private String productSn;
    private String productAttr;
}
