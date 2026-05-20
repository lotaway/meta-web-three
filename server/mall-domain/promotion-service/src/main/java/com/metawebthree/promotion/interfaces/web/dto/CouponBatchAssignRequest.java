package com.metawebthree.promotion.interfaces.web.dto;

import java.util.List;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class CouponBatchAssignRequest {
    @NotNull
    private Long couponTypeId;
    @NotNull
    private List<Long> userIds;
    @NotNull
    private Integer amount;
}
