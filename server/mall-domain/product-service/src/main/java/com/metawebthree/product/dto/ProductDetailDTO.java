package com.metawebthree.product.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "商品详情")
public class ProductDetailDTO {
    @Schema(description = "商品ID")
    private Integer id;
    @Schema(description = "商品实体ID")
    private Integer goodsEntityId;
    @Schema(description = "商品名称")
    private String goodsName;
    @Schema(description = "商品编号")
    private String goodsNo;
    @Schema(description = "商品货号")
    private String goodsArtno;
    @Schema(description = "销量")
    private String saleCount;
    @Schema(description = "售价")
    private BigDecimal salePrice;
    @Schema(description = "市场价")
    private BigDecimal marketPrice;
    @Schema(description = "评分")
    private Double scores;
    @Schema(description = "库存")
    private Integer inventory;
    @Schema(description = "商品图片URL")
    private String imageUrl;
    @Schema(description = "商品图片列表")
    private List<String> pictures;
    @Schema(description = "商品描述")
    private String goodsRemark;
    @Schema(description = "评论数")
    private Integer commentNumber;
    @Schema(description = "评分人数")
    private Integer scoreNumber;
    @Schema(description = "收藏数")
    private Integer favoritesCount;
    @Schema(description = "二维码")
    private String dimensionalCode;
    @Schema(description = "折扣")
    private String goodDiscount;
    @Schema(description = "购买限制")
    private Integer purchase;
    @Schema(description = "是否已收藏")
    private Boolean isFavorites;
    @Schema(description = "规格列表")
    private List<Map<String, Object>> specifications;
    @Schema(description = "属性列表")
    private List<Map<String, Object>> attributes;
    @Schema(description = "面包屑导航")
    private List<Map<String, Object>> breadcrumbs;
    @Schema(description = "评论列表")
    private List<Map<String, Object>> comments;
}
