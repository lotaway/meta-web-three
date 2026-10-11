package com.metawebthree.rma.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ReturnShipping {
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String carrier;
    private String trackingNo;
    private String shippingMethod;
    private String originAddress;
    private String destinationAddress;
    private LocalDateTime shippingDate;
    private LocalDateTime estimatedArrivalDate;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
