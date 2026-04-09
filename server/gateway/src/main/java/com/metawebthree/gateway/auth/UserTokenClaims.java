package com.metawebthree.gateway.auth;

public record UserTokenClaims(Long userId, String userName, String userRole) {
}
