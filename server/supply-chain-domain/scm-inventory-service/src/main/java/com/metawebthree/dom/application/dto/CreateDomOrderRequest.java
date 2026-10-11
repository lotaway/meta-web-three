package com.metawebthree.dom.application.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;
import java.util.List;

@Data
public class CreateDomOrderRequest {
    @NotBlank
    private String originalOrderNo;
    @NotBlank
    private String customerId;
    @NotBlank
    private String customerName;
    private String region;
    private String sourcingStrategy;
    @NotEmpty
    @Valid
    private List<DomOrderItemRequest> items;
}
