package com.metawebthree.promotion.interfaces.web.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class CouponClaimRequest {
    @NotNull
    private Long couponTypeId;
}
