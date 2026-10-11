package com.metawebthree.review.application.dto;

import lombok.Data;

@Data
public class ReviewReplyDTO {
    private Long reviewId;
    private String replyContent;
}