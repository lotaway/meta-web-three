package com.metawebthree.cart.domain;

import java.math.BigDecimal;
import java.util.List;

import lombok.Data;

@Data
public class ProductInfo {
    private Long id;
    private String name;
    private String pic;
    private String subTitle;
    private BigDecimal price;
    private List<String> pictures;
}