package com.metawebthree.notification.domain.model;

public enum NotificationChannel {
    IN_SITE("IN_SITE", "站内信"),
    SMS("SMS", "短信"),
    EMAIL("EMAIL", "邮件"),
    APP("APP", "App推送");

    private final String code;
    private final String desc;

    NotificationChannel(String code, String desc) {
        this.code = code;
        this.desc = desc;
    }

    public String getCode() {
        return code;
    }

    public String getDesc() {
        return desc;
    }
}