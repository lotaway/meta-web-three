package com.metawebthree.review.domain.model;

public enum ReviewStatus {
    PENDING(0, "待审核"),
    APPROVED(1, "已通过"),
    REJECTED(2, "已拒绝"),
    DELETED(3, "已删除");

    private final Integer code;
    private final String desc;

    ReviewStatus(Integer code, String desc) {
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