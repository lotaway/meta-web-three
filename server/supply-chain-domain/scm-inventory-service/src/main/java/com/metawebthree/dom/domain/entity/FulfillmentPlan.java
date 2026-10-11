package com.metawebthree.dom.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class FulfillmentPlan {
    private Long id;
    private Long domOrderId;
    private String domOrderNo;
    private Integer totalLines;
    private Integer fulfilledLines;
    private Integer partiallyFulfilledLines;
    private Integer unfulfilledLines;
    private FulfillmentPlanStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
