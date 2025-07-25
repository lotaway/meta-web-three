package com.metawebthree.common.dto;

import lombok.Data;

@Data
public class ProductDTO {
    private Integer id;
    private String name;
    private String description;
    private Integer[] imageIds;
    private String price;
}