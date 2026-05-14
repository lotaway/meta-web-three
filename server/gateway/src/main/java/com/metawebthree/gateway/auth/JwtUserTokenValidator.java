package com.metawebthree.gateway.auth;

import java.util.Optional;

import org.springframework.stereotype.Component;

import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.utils.UserJwtUtil;

import io.jsonwebtoken.Claims;

@Component
public class JwtUserTokenValidator implements UserTokenValidator {

    private final UserJwtUtil userJwtUtil;
    private final TokenBlacklistService tokenBlacklistService;

    public JwtUserTokenValidator(UserJwtUtil userJwtUtil, TokenBlacklistService tokenBlacklistService) {
        this.userJwtUtil = userJwtUtil;
        this.tokenBlacklistService = tokenBlacklistService;
    }

    @Override
    public UserTokenClaims validate(String token) {
        Optional<Claims> claimsCandidate = userJwtUtil.tryDecode(token);
        if (claimsCandidate.isEmpty()) {
            return null;
        }
        if (tokenBlacklistService.isBlacklisted(token)) {
            return null;
        }
        Claims claims = claimsCandidate.get();
        Long userId = userJwtUtil.getUserId(claims);
        if (userId == null || userJwtUtil.isTokenExpired(claims.getExpiration())) {
            return null;
        }
        return new UserTokenClaims(
                userId,
                userJwtUtil.getUserName(claims),
                userJwtUtil.getUserRole(claims).name());
    }
}
