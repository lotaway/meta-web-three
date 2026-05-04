package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.utils.UserJwtUtil;
import com.metawebthree.user.application.UserService;
import com.metawebthree.user.application.dto.SsoLoginResponseDTO;
import com.metawebthree.user.application.dto.UserDTO;

import lombok.extern.slf4j.Slf4j;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;

import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/sso")
public class SsoController {

    private final UserService userService;
    private final UserJwtUtil jwtUtil;

    @Value("${jwt.tokenHead:Bearer }")
    private String tokenHead;

    public SsoController(UserService userService, UserJwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

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

    @GetMapping("/info")
    public ApiResponse<UserDTO> info(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        UserDTO user = userService.getUserById(userId);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.USER_NOT_FOUND);
        }
        return ApiResponse.success(user);
    }

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

    @GetMapping("/refreshToken")
    public ApiResponse<SsoLoginResponseDTO> refreshToken(@RequestHeader(value = "Authorization", required = false) String authorization) {
        if (authorization == null || authorization.isEmpty()) {
            return ApiResponse.error(ResponseStatus.USER_TOKEN_EXPIRED, "token不能为空");
        }
        
        String token = authorization.replace("Bearer ", "").replace(tokenHead, "");
        
        try {
            String newToken = userService.refreshToken(token);
            if (newToken == null) {
                return ApiResponse.error(ResponseStatus.USER_TOKEN_EXPIRED, "token已经过期");
            }
            return ApiResponse.success(new SsoLoginResponseDTO(newToken, tokenHead));
        } catch (Exception e) {
            log.error("刷新Token失败: {}", e.getMessage());
            return ApiResponse.error(ResponseStatus.USER_TOKEN_EXPIRED, "token刷新失败");
        }
    }
}
