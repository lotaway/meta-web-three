package com.metawebthree.promotion.domain.enums;

public enum PassStatus {
    CLOSED(0),
    OPEN(1);

    private final int code;

    PassStatus(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
