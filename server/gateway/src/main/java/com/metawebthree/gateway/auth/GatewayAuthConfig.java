package com.metawebthree.gateway.auth;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.util.AntPathMatcher;
import org.springframework.util.PathMatcher;

/**
 * Gateway authentication configuration storing role-permission mappings
 * and route-based access control rules.
 */
public class GatewayAuthConfig {

    private final Map<String, Set<String>> rolePermissions = new ConcurrentHashMap<>();
    private final Map<String, List<String>> routeRoles = new ConcurrentHashMap<>();
    private final Map<String, String> routePermissionAliases = new ConcurrentHashMap<>();
    private final PathMatcher pathMatcher = new AntPathMatcher();

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

        // Merchant role - self-service operations for third-party sellers
        rolePermissions.put("MERCHANT", Set.of(
            "product:read", "product:write",
            "order:read", "order:write",
            "inventory:read",
            "promotion:read", "promotion:write",
            "payment:read:own",
            "shop:read", "shop:write"
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
        // Merchant routes
        routeRoles.put("/product-service/merchant/**", List.of("MERCHANT"));
        routeRoles.put("/order-service/merchant/**", List.of("MERCHANT"));
        routeRoles.put("/inventory-service/merchant/**", List.of("MERCHANT"));
        routeRoles.put("/promotion-service/merchant/**", List.of("MERCHANT"));
        routeRoles.put("/payment-service/merchant/**", List.of("MERCHANT"));
        // Tenant routes — public registration excluded at gateway, management ops ADMIN-only,
        // merchant self-service for shop & user association
        routeRoles.put("/tenant-service/tenant/*/approve", List.of("ADMIN"));
        routeRoles.put("/tenant-service/tenant/*/reject", List.of("ADMIN"));
        routeRoles.put("/tenant-service/tenant/*/disable", List.of("ADMIN"));
        routeRoles.put("/tenant-service/tenant/admin/**", List.of("ADMIN"));
        routeRoles.put("/tenant-service/tenant/merchant/**", List.of("MERCHANT"));
        routeRoles.put("/tenant-service/tenant/**", List.of("ADMIN", "MERCHANT"));

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
        String bestMatch = null;
        for (String pattern : routeRoles.keySet()) {
            if (pathMatcher.match(pattern, route)) {
                if (bestMatch == null || pattern.length() > bestMatch.length()) {
                    bestMatch = pattern;
                }
            }
        }
        return bestMatch != null ? bestMatch : route;
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