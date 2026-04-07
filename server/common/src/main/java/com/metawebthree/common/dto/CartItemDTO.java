package com.metawebthree.common.dto;

import io.swagger.v3.oas.annotations.media.Schema;
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
@Schema(description = "购物车项")
public class CartItemDTO {
    @Schema(description = "购物车项ID")
    private Long id;
    @Schema(description = "商品ID")
    private Long productId;
    @Schema(description = "商品SKU ID")
    private Long productSkuId;
    @Schema(description = "会员ID")
    private Long memberId;
    @Schema(description = "数量")
    private Integer quantity;
    @Schema(description = "价格")
    private BigDecimal price;
    @Schema(description = "商品图片")
    private String productPic;
    @Schema(description = "商品名称")
    private String productName;
    @Schema(description = "商品副标题")
    private String productSubTitle;
    @Schema(description = "商品SKU编码")
    private String productSkuCode;
    @Schema(description = "会员昵称")
    private String memberNickname;
    @Schema(description = "创建时间")
    private Date createDate;
    @Schema(description = "修改时间")
    private Date modifyDate;
    @Schema(description = "删除状态")
    private Integer deleteStatus;
    @Schema(description = "商品分类ID")
    private Long productCategoryId;
    @Schema(description = "商品品牌")
    private String productBrand;
    @Schema(description = "商品编号")
    private String productSn;
    @Schema(description = "商品属性")
    private String productAttr;
}
