package com.metawebthree.promotion.domain.enums;

public enum CouponStatus {
    UNUSED(0),
    USED(1);

    private final int code;

    CouponStatus(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    public static CouponStatus fromCode(int code) {
        for (CouponStatus status : values()) {
            if (status.code == code) {
                return status;
            }
        }
        return UNUSED;
    }
}