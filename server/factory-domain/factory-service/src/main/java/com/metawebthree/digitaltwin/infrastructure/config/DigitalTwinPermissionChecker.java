package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.common.annotations.PermissionChecker;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.digitaltwin.common.DigitalTwinPermissions;
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
            DigitalTwinPermissions.DEVICE_READ, DigitalTwinPermissions.DEVICE_CREATE, DigitalTwinPermissions.DEVICE_UPDATE, DigitalTwinPermissions.DEVICE_CONTROL,
            DigitalTwinPermissions.WORKSHOP_READ, DigitalTwinPermissions.WORKSHOP_CREATE, DigitalTwinPermissions.WORKSHOP_UPDATE,
            DigitalTwinPermissions.PRODUCTION_LINE_READ, DigitalTwinPermissions.PRODUCTION_LINE_CREATE, DigitalTwinPermissions.PRODUCTION_LINE_UPDATE,
            DigitalTwinPermissions.ALERT_READ, DigitalTwinPermissions.ALERT_CREATE, DigitalTwinPermissions.ALERT_ACK, DigitalTwinPermissions.ALERT_RESOLVE,
            DigitalTwinPermissions.STATS_READ
        ),
        "OPERATOR", Set.of(
            DigitalTwinPermissions.DEVICE_READ, DigitalTwinPermissions.DEVICE_CREATE, DigitalTwinPermissions.DEVICE_UPDATE, DigitalTwinPermissions.DEVICE_CONTROL,
            DigitalTwinPermissions.WORKSHOP_READ, DigitalTwinPermissions.WORKSHOP_CREATE, DigitalTwinPermissions.WORKSHOP_UPDATE,
            DigitalTwinPermissions.PRODUCTION_LINE_READ, DigitalTwinPermissions.PRODUCTION_LINE_CREATE, DigitalTwinPermissions.PRODUCTION_LINE_UPDATE,
            DigitalTwinPermissions.ALERT_READ, DigitalTwinPermissions.ALERT_CREATE, DigitalTwinPermissions.ALERT_ACK, DigitalTwinPermissions.ALERT_RESOLVE,
            DigitalTwinPermissions.STATS_READ
        ),
        "VIEWER", Set.of(
            DigitalTwinPermissions.DEVICE_READ,
            DigitalTwinPermissions.WORKSHOP_READ,
            DigitalTwinPermissions.PRODUCTION_LINE_READ,
            DigitalTwinPermissions.ALERT_READ,
            DigitalTwinPermissions.STATS_READ
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
