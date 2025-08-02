package com.metawebthree.common.utils;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.JwtParser;
import io.jsonwebtoken.JwtParserBuilder;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.UnsupportedJwtException;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.security.Key;
import java.security.SignatureException;
import java.util.Date;
import java.util.Map;
import java.util.Optional;

import javax.crypto.spec.SecretKeySpec;

@Component
public class JwtUtil {

    @Value("${jwt.secret:123123}")
    protected String secret;

    @Value("${name:JWT}")
    protected String name;

    protected String secName = "JWT";

    private final Long ONE_MONTH = 30L * 24 * 60 * 60 * 1000;

    public String generate(String subject) {
        return Jwts.builder()
                .setHeaderParam(name, secName)
                .setSubject(subject)
                .setIssuedAt(new Date())
                .signWith(getSignKey())
                .compact();
    }

    public String generate(String subject, Map<String, Object> claimsMap) {
        return generate(subject, claimsMap, getDefaultExpiration());
    }

    public String generate(String subject, Map<String, Object> claimsMap, Date expiration) {
        return Jwts.builder()
                .setHeaderParam(name, secName)
                .setSubject(subject)
                .setClaims(claimsMap)
                .setIssuedAt(new Date())
                .setExpiration(expiration)
                .signWith(getSignKey())
                .compact();
    }

    public Date getDefaultExpiration() {
        return new Date(System.currentTimeMillis() + ONE_MONTH);
    }

    protected Key getSignKey() {
        return new SecretKeySpec(secret.getBytes(), SignatureAlgorithm.ES512.getJcaName());
    }

    public Claims decode(String token) throws ExpiredJwtException, UnsupportedJwtException, MalformedJwtException,
            SignatureException, IllegalArgumentException {
        JwtParserBuilder builder = Jwts.parserBuilder();
        builder.setSigningKey(getSignKey());
        JwtParser parser = builder.build();
        return parser.parseClaimsJws(token).getBody();
    }

    public Optional<Claims> tryDecode(String token) {
        try {
            return Optional.of(decode(token));
        } catch (Throwable e) {
            return Optional.empty();
        }
    }

    public boolean isTokenExpired(Date expiration) {
        return expiration.before(new Date());
    }

    public boolean isTokenNotExpired(Date expiration) {
        return !isTokenExpired(expiration);
    }

    public boolean isTokenValid(Claims claims) {
        return this.isTokenNotExpired(claims.getExpiration());
    }

}
