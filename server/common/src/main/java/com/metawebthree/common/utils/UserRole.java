package com.metawebthree.common.utils;

public enum UserRole {
    USER(0b00001L),
    SHOP(0b00010L),
    CUSTOM_SERVICE(0b00100L),
    ADMIN(0b01000L);

    private final Long value;

    UserRole(Long value) {
        this.value = value;
    }
}
