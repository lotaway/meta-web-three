package com.metawebthree.notification.domain.model;

public enum SendStatus {
    PENDING(0, "待发送"),
    SENDING(1, "发送中"),
    SENT(2, "已发送"),
    FAILED(3, "发送失败");

    private final Integer code;
    private final String desc;

    SendStatus(Integer code, String desc) {
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