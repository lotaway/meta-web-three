package com.metawebthree.product.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Schema(description = "商品信息")
public class ProductDTO {
    @Schema(description = "商品ID")
    private Integer id;
    @Schema(description = "商品编号")
    private String goodsNo;
    @Schema(description = "商品名称")
    private String name;
    @Schema(description = "商品图片URL")
    private String imageUrl;
    @Schema(description = "售价")
    private String price;
    @Schema(description = "市场价")
    private String marketPrice;
    @Schema(description = "评分")
    private Double scores;
    @Schema(description = "销量")
    private Integer saleCount;
    @Schema(description = "评论数")
    private Integer commentNumber;
}
