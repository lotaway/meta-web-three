package com.metawebthree.common.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ProductDetailDTO {
    private Integer id;
    private Integer goodsEntityId;
    private String goodsName;
    private String goodsNo;
    private String goodsArtno;
    private String saleCount;
    private BigDecimal salePrice;
    private BigDecimal marketPrice;
    private Double scores;
    private Integer inventory;
    private String imageUrl;
    private List<String> pictures;
    private String goodsRemark;
    private Integer commentNumber;
    private Integer scoreNumber;
    private Integer favoritesCount;
    private String dimensionalCode;
    private String goodDiscount;
    private Integer purchase;
    private Boolean isFavorites;
    private List<Map<String, Object>> specifications;
    private List<Map<String, Object>> attributes;
    private List<Map<String, Object>> breadcrumbs;
    private List<Map<String, Object>> comments;
}
