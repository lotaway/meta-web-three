package com.metawebthree.utils;

import io.jsonwebtoken.Claims;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import com.metawebthree.common.utils.JwtUtil;

@Component
public class UserJwtUtil extends JwtUtil {

    public String generate(Long userId) {
        return generate(userId.toString(), generateClaimsMap(userId.toString()));
    }

    public Map<String, Object> generateClaimsMap(String name) {
        return generateClaimsMap(name, UserRole.USER);
    }

    public Map<String, Object> generateClaimsMap(String name, UserRole role) {
        Map<String, Object> claimsMap = new HashMap<>();
        claimsMap.put("name", name);
        claimsMap.put("role", role.name());
        return claimsMap;
    }

    public Optional<Long> getUserId(Claims claims) {
        return Optional.of(Long.parseLong(claims.getSubject()));
    }

}
