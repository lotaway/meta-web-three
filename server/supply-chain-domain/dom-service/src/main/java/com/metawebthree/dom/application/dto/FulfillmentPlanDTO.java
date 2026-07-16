package com.metawebthree.dom.application.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class FulfillmentPlanDTO {
    private Long id;
    private Long domOrderId;
    private String domOrderNo;
    private Integer totalLines;
    private Integer fulfilledLines;
    private Integer partiallyFulfilledLines;
    private Integer unfulfilledLines;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
