package com.metawebthree.review.application.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ReviewDTO {
    private Long id;
    private Long orderId;
    private Long orderItemId;
    private Long productId;
    private Long skuId;
    private Long userId;
    private String userNickname;
    private String userAvatar;
    private Long storeId;
    private String storeName;
    private Integer rating;
    private String content;
    private String images;
    private Integer status;
    private String statusDesc;
    private Integer likeCount;
    private Integer replyCount;
    private String replyContent;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}