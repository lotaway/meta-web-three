package com.metawebthree.common.annotations;

public interface PermissionChecker {
    boolean hasPermission(Long userId, String permission);
}
