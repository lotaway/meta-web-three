package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.utils.UserJwtUtil;
import com.metawebthree.user.application.PasskeyService;
import com.metawebthree.user.application.dto.LoginResponseDTO;
import com.metawebthree.user.application.dto.UserDTO;
import com.metawebthree.user.application.UserService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;

import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/user/passkey")
@Tag(name = "Passkey", description = "Passkey 注册与认证")
public class PasskeyController {

    private final PasskeyService passkeyService;
    private final UserService userService;
    private final UserJwtUtil jwtUtil;

    public PasskeyController(PasskeyService passkeyService, UserService userService, UserJwtUtil jwtUtil) {
        this.passkeyService = passkeyService;
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

    @PostMapping("/register/options")
    @Operation(summary = "生成注册选项", description = "获取 Passkey 注册所需的配置选项")
    public ApiResponse<Map<String, Object>> generateRegistrationOptions(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam String rpId) {
        Map<String, Object> options = passkeyService.generateRegistrationOptions(userId, rpId);
        return ApiResponse.success(options);
    }

    @PostMapping("/register/verify")
    @Operation(summary = "验证并存储凭证", description = "验证客户端传回的 attestation 并存储凭证")
    public ApiResponse<Void> verifyRegistration(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam String rpId,
            @RequestBody Map<String, Object> attestation) {
        passkeyService.verifyAndStoreCredential(userId, rpId, attestation);
        return ApiResponse.success();
    }

    @PostMapping("/login/options")
    @Operation(summary = "生成认证选项", description = "获取 Passkey 认证所需的 challenge")
    public ApiResponse<Map<String, Object>> generateAuthenticationOptions(
            @RequestParam String rpId) {
        Map<String, Object> options = passkeyService.generateAuthenticationOptions(rpId);
        return ApiResponse.success(options);
    }

    @PostMapping("/login/verify")
    @Operation(summary = "验证认证并登录", description = "验证客户端传回的 assertion 并返回 JWT")
    public ApiResponse<LoginResponseDTO> verifyAuthentication(
            @RequestParam String rpId,
            @RequestBody Map<String, Object> assertion) {
        Long userId = passkeyService.verifyAndAuthenticate(rpId, assertion);
        UserDTO user = userService.getUserById(userId);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.USER_NOT_FOUND);
        }

        Map<String, Object> claims = new java.util.HashMap<>();
        claims.put("userId", user.getId());
        claims.put("name", user.getNickname());
        claims.put("role", "USER");

        String token = jwtUtil.generate(user.getId().toString(), claims);
        return ApiResponse.success(new LoginResponseDTO(token, user, null, "passkey"));
    }

    @GetMapping("/list")
    @Operation(summary = "获取 Passkey 列表", description = "获取用户已注���的 Passkey 凭证列表")
    public ApiResponse<List<Map<String, Object>>> getPasskeyList(
            @RequestHeader(HeaderConstants.USER_ID) Long userId) {
        List<Map<String, Object>> list = passkeyService.getPasskeyList(userId);
        return ApiResponse.success(list);
    }

    @DeleteMapping("/delete")
    @Operation(summary = "删除 Passkey", description = "删除指定的 Passkey 凭证")
    public ApiResponse<Void> deletePasskey(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam String credentialId) {
        boolean result = passkeyService.deletePasskey(userId, credentialId);
        if (result) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.PARAM_TYPE_ERROR, "Credential not found");
    }
}
