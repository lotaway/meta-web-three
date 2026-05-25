package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.common.annotations.PermissionChecker;
import com.metawebthree.common.constants.HeaderConstants;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.Map;
import java.util.Set;

@Component
public class DigitalTwinPermissionChecker implements PermissionChecker {

    private static final Map<String, Set<String>> ROLE_PERMISSIONS = Map.of(
        "ADMIN", Set.of(
            "dt:device:read", "dt:device:create", "dt:device:update", "dt:device:control",
            "dt:workshop:read", "dt:workshop:create", "dt:workshop:update",
            "dt:production-line:read", "dt:production-line:create", "dt:production-line:update",
            "dt:alert:read", "dt:alert:create", "dt:alert:ack", "dt:alert:resolve",
            "dt:stats:read"
        ),
        "ADMIN", Set.of(
            "dt:device:read", "dt:device:create", "dt:device:update", "dt:device:control",
            "dt:workshop:read", "dt:workshop:create", "dt:workshop:update",
            "dt:production-line:read", "dt:production-line:create", "dt:production-line:update",
            "dt:alert:read", "dt:alert:create", "dt:alert:ack", "dt:alert:resolve",
            "dt:stats:read"
        ),
        "VIEWER", Set.of(
            "dt:device:read",
            "dt:workshop:read",
            "dt:production-line:read",
            "dt:alert:read",
            "dt:stats:read"
        )
    );

    @Override
    public boolean hasPermission(Long userId, String permission) {
        String role = getRoleFromRequest();
        if (role == null) {
            return false;
        }
        Set<String> permissions = ROLE_PERMISSIONS.get(role);
        return permissions != null && permissions.contains(permission);
    }

    private String getRoleFromRequest() {
        ServletRequestAttributes attrs = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attrs == null) {
            return null;
        }
        HttpServletRequest request = attrs.getRequest();
        String role = request.getHeader(HeaderConstants.USER_ROLE);
        return role != null ? role.toUpperCase() : null;
    }
}
