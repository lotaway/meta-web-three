package com.metawebthree.review.domain.model;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ReviewDO {
    private Long id;
    private Long orderId;
    private Long orderItemId;
    private Long productId;
    private Long skuId;
    private Long userId;
    private Long storeId;
    private Integer rating;
    private String content;
    private String images;
    private Integer status;
    private Integer likeCount;
    private Integer replyCount;
    private String replyContent;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}