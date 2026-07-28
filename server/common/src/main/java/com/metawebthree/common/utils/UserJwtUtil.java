package com.metawebthree.common.utils;

import io.jsonwebtoken.Claims;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class UserJwtUtil extends JwtUtil {

    private final static String USER_NAME_KEY = "name";
    private final static String USER_ROLE_KEY = "role";
    private final static String TENANT_ID_KEY = "tenantId";

    public String generate(Long userId) {
        return generate(userId.toString(), generateClaimsMap(userId.toString()));
    }

    public Map<String, Object> generateClaimsMap(String name) {
        return generateClaimsMap(name, UserRole.USER, null);
    }

    public Map<String, Object> generateClaimsMap(String name, UserRole role) {
        return generateClaimsMap(name, role, null);
    }

    public Map<String, Object> generateClaimsMap(String name, UserRole role, Long tenantId) {
        Map<String, Object> claimsMap = new HashMap<>();
        claimsMap.put(USER_NAME_KEY, name);
        claimsMap.put(USER_ROLE_KEY, role.name());
        if (tenantId != null) {
            claimsMap.put(TENANT_ID_KEY, tenantId);
        }
        return claimsMap;
    }

    public Long getUserId(Claims claims) {
        return Long.parseLong(claims.getSubject());
    }

    public String getUserName(Claims claims) {
        return claims.get(USER_NAME_KEY, String.class);
    }

    public UserRole getUserRole(Claims claims) {
        return UserRole.valueOf(claims.get(USER_ROLE_KEY, String.class));
    }

    public Long getTenantId(Claims claims) {
        return claims.get(TENANT_ID_KEY, Long.class);
    }

}
