package com.metawebthree.promotion.interfaces.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public class CouponAssignRequest {
    @NotBlank
    private String code;
    @NotNull
    private Long userId;

    public String getCode() { return code; }
    public void setCode(String code) { this.code = code; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
}
