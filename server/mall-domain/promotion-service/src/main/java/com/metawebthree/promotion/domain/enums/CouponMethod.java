package com.metawebthree.promotion.domain.enums;

public enum CouponMethod {
    ADMIN_ASSIGN(0),
    SELF_CLAIM(1),
    TRANSFER(2);

    private final int code;

    CouponMethod(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
