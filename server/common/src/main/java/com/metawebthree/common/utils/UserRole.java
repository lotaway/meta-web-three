package com.metawebthree.common.utils;

import java.util.Optional;

public enum UserRole {
    USER(0b00000001L),
    SHOP(0b00000010L),
    CUSTOM_SERVICE(0b00000100L),
    ADMIN(0b00001000L),
    APPLICATION(0b00010000L);

    private final Long value;

    UserRole(Long value) {
        this.value = value;
    }

    public Long getValue() {
        return value;
    }

    public static UserRole valueOf(Long value) {
        for (UserRole role : UserRole.values()) {
            if (role.getValue().equals(value)) {
                return role;
            }
        }
        throw new IllegalArgumentException("Unknown UserRole value: " + value);
    }

    public static Optional<UserRole> tryValueOf(Long value) {
        for (UserRole role : UserRole.values()) {
            if (role.getValue().equals(value)) {
                return Optional.of(role);
            }
        }
        return Optional.empty();
    }
}
