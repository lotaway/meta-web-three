package com.metawebthree.dom.application.dto;

import lombok.Data;
import java.util.List;

@Data
public class CreateDomOrderRequest {
    private String originalOrderNo;
    private String customerId;
    private String customerName;
    private String region;
    private String sourcingStrategy;
    private List<DomOrderItemRequest> items;
}
