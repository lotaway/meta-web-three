package com.metawebthree.promotion.interfaces.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class CouponGenerateRequest {
    @NotBlank
    private String batchId;
    @NotNull
    private Integer count;
}
