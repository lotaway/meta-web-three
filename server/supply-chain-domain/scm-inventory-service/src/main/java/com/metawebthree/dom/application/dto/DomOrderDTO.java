package com.metawebthree.dom.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class DomOrderDTO {
    private Long id;
    private String domOrderNo;
    private String originalOrderNo;
    private String customerId;
    private String customerName;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private Integer priority;
    private String sourcingStrategy;
    private String region;
    private List<DomOrderLineDTO> lines;
    private FulfillmentPlanDTO fulfillmentPlan;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
