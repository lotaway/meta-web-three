package com.metawebthree.common.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ProductDTO {
    private Integer id;
    private String goodsNo;
    private String name;
    private String imageUrl;
    private String price;
    private String marketPrice;
    private Double scores;
    private Integer saleCount;
    private Integer commentNumber;
}