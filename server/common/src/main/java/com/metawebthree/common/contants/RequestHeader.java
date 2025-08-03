package com.metawebthree.common.contants;

public enum RequestHeader {
    USER_ID("X-User-Id"),
    USER_ROLE("X-User-Role"),
    USER_NAME("X-User-Name");

    private final String value;

    RequestHeader(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
