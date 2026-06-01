package com.metawebthree.notification.domain.model;

public enum ReadStatus {
    UNREAD(0, "未读"),
    READ(1, "已读");

    private final Integer code;
    private final String desc;

    ReadStatus(Integer code, String desc) {
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