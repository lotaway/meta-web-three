package com.metawebthree.aftersale.domain.model;

public enum AfterSaleStatus {
    PENDING(1, "待处理"),
    PROCESSING(2, "处理中"),
    APPROVED(3, "已批准"),
    REJECTED(4, "已拒绝"),
    COMPLETED(5, "已完成"),
    CANCELLED(6, "已取消");

    private final Integer code;
    private final String desc;

    AfterSaleStatus(Integer code, String desc) {
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