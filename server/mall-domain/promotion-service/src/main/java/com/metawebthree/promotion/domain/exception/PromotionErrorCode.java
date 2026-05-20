package com.metawebthree.promotion.domain.exception;

public enum PromotionErrorCode {
    INVALID_REQUEST("PROMO-1001"),
    NOT_FOUND("PROMO-1002"),
    NOT_ALLOWED("PROMO-1003"),
    CONFLICT("PROMO-1004"),
    EXPIRED("PROMO-1005");

    private final String code;

    PromotionErrorCode(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}
