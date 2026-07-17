package com.metawebthree.dom.application.dto;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import java.math.BigDecimal;

@Data
public class DomOrderItemRequest {
    @NotBlank
    private String skuCode;
    private String skuName;
    @NotNull
    @Min(1)
    private Integer quantity;
    @NotNull
    @Min(0)
    private BigDecimal unitPrice;
}
