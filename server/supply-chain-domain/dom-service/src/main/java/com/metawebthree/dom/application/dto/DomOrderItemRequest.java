package com.metawebthree.dom.application.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class DomOrderItemRequest {
    private String skuCode;
    private String skuName;
    private Integer quantity;
    private BigDecimal unitPrice;
}
