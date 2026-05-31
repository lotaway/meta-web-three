package com.metawebthree.gateway.auth;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Gateway authentication configuration storing role-permission mappings
 * and route-based access control rules.
 */
public class GatewayAuthConfig {

    private final Map<String, Set<String>> rolePermissions = new ConcurrentHashMap<>();
    private final Map<String, List<String>> routeRoles = new ConcurrentHashMap<>();
    private final Map<String, String> routePermissionAliases = new ConcurrentHashMap<>();

    public GatewayAuthConfig() {
        initializeDefaultPermissions();
    }

    private void initializeDefaultPermissions() {
        // Admin role - full access
        rolePermissions.put("ADMIN", Set.of("*"));

        // Manager role - management operations
        rolePermissions.put("MANAGER", Set.of(
            "order:read", "order:write",
            "product:read", "product:write",
            "inventory:read", "inventory:write",
            "user:read",
            "payment:read",
            "report:read"
        ));

        // Regular user role - basic operations
        rolePermissions.put("USER", Set.of(
            "order:read", "order:write",
            "product:read",
            "inventory:read",
            "user:read:own",
            "payment:read:own"
        ));

        // Guest role - read-only public access
        rolePermissions.put("GUEST", Set.of(
            "product:read:public",
            "flash:read"
        ));

        // Define route-to-role mappings
        routeRoles.put("/user-service/admin/**", List.of("ADMIN"));
        routeRoles.put("/user-service/role/**", List.of("ADMIN"));
        routeRoles.put("/order-service/admin/**", List.of("ADMIN", "MANAGER"));
        routeRoles.put("/order-service/**", List.of("ADMIN", "MANAGER", "USER"));
        routeRoles.put("/product-service/admin/**", List.of("ADMIN", "MANAGER"));
        routeRoles.put("/product-service/**", List.of("ADMIN", "MANAGER", "USER"));
        routeRoles.put("/inventory-service/**", List.of("ADMIN", "MANAGER", "USER"));
        routeRoles.put("/payment-service/**", List.of("ADMIN", "MANAGER", "USER"));
        routeRoles.put("/wallet-service/**", List.of("ADMIN", "MANAGER", "USER"));
        routeRoles.put("/promotion-service/admin/**", List.of("ADMIN", "MANAGER"));
        routeRoles.put("/promotion-service/**", List.of("ADMIN", "MANAGER", "USER", "GUEST"));

        // Permission aliases for backward compatibility
        routePermissionAliases.put("order:read:own", "order:read");
        routePermissionAliases.put("payment:read:own", "payment:read");
        routePermissionAliases.put("user:read:own", "user:read");
    }

    public boolean hasPermission(String role, String requiredPermission) {
        Set<String> permissions = rolePermissions.get(role);
        if (permissions == null) {
            return false;
        }
        // Admin has full access
        if (permissions.contains("*")) {
            return true;
        }
        // Check exact permission or wildcard
        return permissions.contains(requiredPermission) 
            || permissions.contains(requiredPermission.split(":")[0] + ":*");
    }

    public boolean hasRoleForRoute(String role, String route) {
        List<String> allowedRoles = routeRoles.get(matchRoute(route));
        if (allowedRoles == null || allowedRoles.isEmpty()) {
            return true; // No role restriction for this route
        }
        return allowedRoles.contains(role) || allowedRoles.contains("*");
    }

    private String matchRoute(String route) {
        // Find the most specific route match
        for (String pattern : routeRoles.keySet()) {
            if (matchPath(pattern, route)) {
                return pattern;
            }
        }
        return route;
    }

    private boolean matchPath(String pattern, String path) {
        if (pattern.endsWith("/**")) {
            String prefix = pattern.substring(0, pattern.length() - 3);
            return path.startsWith(prefix);
        }
        return pattern.equals(path);
    }

    public Set<String> getPermissions(String role) {
        return rolePermissions.getOrDefault(role, Set.of());
    }

    public List<String> getRouteRoles(String route) {
        return routeRoles.get(matchRoute(route));
    }

    public void addRolePermission(String role, String permission) {
        rolePermissions.computeIfAbsent(role, k -> ConcurrentHashMap.newKeySet()).add(permission);
    }

    public void addRouteRole(String route, String role) {
        routeRoles.computeIfAbsent(route, k -> new ArrayList<>()).add(role);
    }
}