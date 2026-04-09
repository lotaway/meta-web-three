package com.metawebthree.gateway.auth;

public interface UserTokenValidator {

    UserTokenClaims validate(String token);
}
