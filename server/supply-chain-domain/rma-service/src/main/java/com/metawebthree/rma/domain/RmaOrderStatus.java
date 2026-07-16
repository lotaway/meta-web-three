package com.metawebthree.rma.domain;

public enum RmaOrderStatus {
    PENDING(0),
    AWAITING_INSPECTION(1),
    INSPECTED(2),
    AWAITING_DISPOSITION(3),
    DISPOSED(4),
    COMPLETED(5),
    CANCELLED(6);

    private final int code;

    RmaOrderStatus(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
