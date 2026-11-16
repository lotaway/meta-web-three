package com.metawebthree.cs.infrastructure.client;

import java.util.Optional;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.GetUserPhoneRequest;
import com.metawebthree.common.generated.rpc.GetUserPhoneResponse;
import com.metawebthree.common.generated.rpc.GetUserProfileRequest;
import com.metawebthree.common.generated.rpc.GetUserProfileResponse;
import com.metawebthree.common.generated.rpc.UserInfoProto;
import com.metawebthree.common.generated.rpc.UserService;
import com.metawebthree.cs.domain.ports.UserQueryPort;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class UserQueryPortImpl implements UserQueryPort {

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    @Override
    public Optional<String> findNickname(Long userId) {
        UserInfoProto profile = getUserProfile(userId);
        return profile != null && !profile.getNickname().isEmpty()
                ? Optional.of(profile.getNickname())
                : Optional.empty();
    }

    @Override
    public Optional<String> findAvatar(Long userId) {
        UserInfoProto profile = getUserProfile(userId);
        return profile != null && !profile.getAvatar().isEmpty()
                ? Optional.of(profile.getAvatar())
                : Optional.empty();
    }

    @Override
    public Optional<String> findPhone(Long userId) {
        try {
            GetUserPhoneResponse response = userService.getUserPhone(
                    GetUserPhoneRequest.newBuilder().setUserId(userId).build());
            if (response != null && response.getSuccess() && !response.getPhone().isEmpty()) {
                return Optional.of(response.getPhone());
            }
        } catch (Exception e) {
            log.warn("Failed to fetch phone for user {} via RPC: {}", userId, e.getMessage());
        }
        return Optional.empty();
    }

    private UserInfoProto getUserProfile(Long userId) {
        if (userId == null) {
            return null;
        }
        try {
            GetUserProfileResponse response = userService.getUserProfile(
                    GetUserProfileRequest.newBuilder().setUserId(userId).build());
            if (response != null && response.getSuccess() && response.hasUser()) {
                return response.getUser();
            }
        } catch (Exception e) {
            log.warn("Failed to fetch profile for user {} via RPC: {}", userId, e.getMessage());
        }
        return null;
    }
}
