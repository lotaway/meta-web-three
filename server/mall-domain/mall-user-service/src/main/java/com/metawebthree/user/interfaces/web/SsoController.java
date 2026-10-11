package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.utils.UserJwtUtil;
import com.metawebthree.user.application.UserService;
import com.metawebthree.user.application.dto.SsoLoginResponseDTO;
import com.metawebthree.user.application.dto.UserDTO;

import io.jsonwebtoken.Claims;
import lombok.extern.slf4j.Slf4j;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/sso")
@Tag(name = "Sso Controller", description = "Authentication and Session Management")
public class SsoController {

    private final UserService userService;
    private final UserJwtUtil jwtUtil;
    private final TokenBlacklistService tokenBlacklistService;

    @Value("${jwt.tokenHead:Bearer }")
    private String tokenHead;

    public SsoController(UserService userService, UserJwtUtil jwtUtil, TokenBlacklistService tokenBlacklistService) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
        this.tokenBlacklistService = tokenBlacklistService;
    }

    @Operation(summary = "User Login", description = "Authenticate user and return session information")
    @PostMapping("/login")
    public ApiResponse<SsoLoginResponseDTO> login(@RequestParam String username,
                                                   @RequestParam String password) throws NoSuchAlgorithmException {
        UserDTO user = userService.validateUser(username, password, null);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.USER_PASSWORD_ERROR, "用户名或密码错误");
        }

        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("name", user.getNickname() != null ? user.getNickname() : user.getUsername());
        claims.put("role", "USER");

        String token = jwtUtil.generate(user.getId().toString(), claims);

        return ApiResponse.success(new SsoLoginResponseDTO(token, tokenHead));
    }

    @Operation(summary = "User Registration", description = "Create a new user account")
    @PostMapping("/register")
    public ApiResponse<Void> register(@RequestParam String username,
                                       @RequestParam String password,
                                       @RequestParam(required = false) String telephone,
                                       @RequestParam(required = false) String authCode) {
        try {
            String email = telephone != null ? telephone : username;
            userService.createUser(email, password);
            return ApiResponse.success();
        } catch (Exception e) {
            log.error("注册失败: {}", e.getMessage());
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR, "注册失败");
        }
    }

    @Operation(summary = "Get User Info", description = "Get current authenticated user information")
    @GetMapping("/info")
    public ApiResponse<UserDTO> info(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        UserDTO user = userService.getUserById(userId);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.USER_NOT_FOUND);
        }
        return ApiResponse.success(user);
    }

    @Operation(summary = "Get Auth Code", description = "Generate and send an authentication code to the specified telephone number")
    @GetMapping("/getAuthCode")
    public ApiResponse<String> getAuthCode(@RequestParam String telephone) {
        try {
            String authCode = userService.generateAuthCode(telephone);
            return ApiResponse.success(authCode);
        } catch (Exception e) {
            log.error("获取验证码失败: {}", e.getMessage());
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR, "获取验证码失败");
        }
    }

    @Operation(summary = "Login by Phone", description = "Authenticate user using telephone number and authentication code")
    @PostMapping("/loginByPhone")
    public ApiResponse<SsoLoginResponseDTO> loginByPhone(@RequestParam String telephone,
                                                           @RequestParam String authCode) throws NoSuchAlgorithmException {
        UserDTO user = userService.validateUserByPhone(telephone, authCode);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.USER_PASSWORD_ERROR, "手机号或验证码错误");
        }

        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("name", user.getNickname() != null ? user.getNickname() : user.getUsername());
        claims.put("role", "USER");

        String token = jwtUtil.generate(user.getId().toString(), claims);

        return ApiResponse.success(new SsoLoginResponseDTO(token, tokenHead));
    }

    @Operation(summary = "User Logout", description = "Logout and invalidate the current token")
    @PostMapping("/logout")
    public ApiResponse<Void> logout(@RequestHeader("X-Original-Token") String originalToken) {
        try {
            String token = originalToken.replace("Bearer ", "").replace(tokenHead, "");
            Claims claims = jwtUtil.tryDecode(token).orElse(null);
            if (claims != null && claims.getExpiration() != null) {
                long ttl = (claims.getExpiration().getTime() - System.currentTimeMillis()) / 1000;
                if (ttl > 0) {
                    tokenBlacklistService.blacklist(originalToken, ttl);
                }
            }
        } catch (Exception e) {
            log.warn("Logout blacklist failed: {}", e.getMessage());
        }
        return ApiResponse.success();
    }

    @Operation(summary = "Update Password", description = "Update user password using telephone number and authentication code")
    @PostMapping("/updatePassword")
    public ApiResponse<Void> updatePassword(@RequestParam String telephone,
                                             @RequestParam String password,
                                             @RequestParam String authCode) {
        try {
            userService.updatePassword(telephone, password, authCode);
            return ApiResponse.success();
        } catch (Exception e) {
            log.error("修改密码失败: {}", e.getMessage());
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR, "密码修改失败");
        }
    }

    @Operation(summary = "Refresh Token", description = "Refresh the authentication token")
    @PostMapping("/refreshToken")
    public ApiResponse<SsoLoginResponseDTO> refreshToken(@RequestHeader("X-Original-Token") String originalToken) {
        try {
            String token = originalToken.replace("Bearer ", "").replace(tokenHead, "");
            if (tokenBlacklistService.isBlacklisted(token)) {
                return ApiResponse.error(ResponseStatus.USER_TOKEN_EXPIRED, "token已失效");
            }
            Claims claims = jwtUtil.tryDecode(token).orElse(null);
            if (claims == null || jwtUtil.isTokenExpired(claims.getExpiration())) {
                return ApiResponse.error(ResponseStatus.USER_TOKEN_EXPIRED, "token已过期");
            }
            String newToken = jwtUtil.generate(claims.getSubject(), claims);
            long ttl = Math.max(1, (claims.getExpiration().getTime() - System.currentTimeMillis()) / 1000);
            tokenBlacklistService.blacklist(token, ttl);
            return ApiResponse.success(new SsoLoginResponseDTO(newToken, tokenHead));
        } catch (Exception e) {
            log.error("刷新Token失败: {}", e.getMessage());
            return ApiResponse.error(ResponseStatus.USER_TOKEN_EXPIRED, "token刷新失败");
        }
    }
}
