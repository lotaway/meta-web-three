package com.metawebthree.inventoryalert.domain.model;

public enum AlertStatus {
    PENDING(1, "待处理"),
    NOTIFIED(2, "已通知"),
    PROCESSING(3, "处理中"),
    RESOLVED(4, "已解决"),
    IGNORED(5, "已忽略");

    private final Integer code;
    private final String desc;

    AlertStatus(Integer code, String desc) {
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