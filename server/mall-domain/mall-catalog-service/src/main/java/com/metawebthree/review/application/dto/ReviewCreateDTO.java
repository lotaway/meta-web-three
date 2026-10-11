package com.metawebthree.review.application.dto;

import lombok.Data;

@Data
public class ReviewCreateDTO {
    private Long orderId;
    private Long orderItemId;
    private Long productId;
    private Long skuId;
    private Integer rating;
    private String content;
    private String images;
}