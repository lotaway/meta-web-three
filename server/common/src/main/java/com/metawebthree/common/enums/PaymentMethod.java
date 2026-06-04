package com.metawebthree.common.enums;

public enum PaymentMethod {
    WECHAT(1, "WeChat Pay"),
    ALIPAY(2, "Alipay"),
    CARD(3, "Card Payment"),
    OTHER(4, "Other");

    private final int code;
    private final String displayName;

    PaymentMethod(int code, String displayName) {
        this.code = code;
        this.displayName = displayName;
    }

    public int getCode() { return code; }
    public String getDisplayName() { return displayName; }

    public static PaymentMethod fromCode(int code) {
        for (PaymentMethod m : values()) {
            if (m.code == code) return m;
        }
        return OTHER;
    }
}
