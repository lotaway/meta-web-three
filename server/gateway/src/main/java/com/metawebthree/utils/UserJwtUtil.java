package com.metawebthree.utils;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.JwtParser;
import io.jsonwebtoken.JwtParserBuilder;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.UnsupportedJwtException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.security.Key;
import java.security.SignatureException;
import java.util.Date;
import java.util.Optional;

import javax.crypto.spec.SecretKeySpec;
import com.metawebthree.common.utils.JwtUtil;

@Component
public class UserJwtUtil extends JwtUtil {

    // public String generate(Long userId) {
    //     return generate(userId.toString());
    // }

    public Optional<Long> getUserId(Claims claims) {
        return Optional.of(Long.parseLong(claims.getSubject()));
    }

}
