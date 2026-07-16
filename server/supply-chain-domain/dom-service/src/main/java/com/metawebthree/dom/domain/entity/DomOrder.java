package com.metawebthree.dom.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class DomOrder {
    private Long id;
    private String domOrderNo;
    private String originalOrderNo;
    private String customerId;
    private String customerName;
    private DomOrderStatus status;
    private BigDecimal totalAmount;
    private String currency;
    private Integer priority;
    private SourcingStrategy sourcingStrategy;
    private String region;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}
