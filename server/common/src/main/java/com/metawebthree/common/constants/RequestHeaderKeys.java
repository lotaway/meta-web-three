package com.metawebthree.common.constants;

public enum RequestHeaderKeys {
    USER_ID(HeaderConstants.USER_ID),
    USER_ROLE(HeaderConstants.USER_ROLE),
    USER_NAME(HeaderConstants.USER_NAME),
    TENANT_ID(HeaderConstants.TENANT_ID);

    private final String value;

    RequestHeaderKeys(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
