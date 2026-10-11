package com.metawebthree.aftersale.domain.model;

public enum AfterSaleType {
    RETURN_GOODS(1, "退货退款"),
    EXCHANGE_GOODS(2, "换货"),
    REFUND_ONLY(3, "仅退款");

    private final Integer code;
    private final String desc;

    AfterSaleType(Integer code, String desc) {
        this.code = code;
        this.desc = desc;
    }

    public Integer getCode() {
        return code;
    }

    public String getDesc() {
        return desc;
    }
}