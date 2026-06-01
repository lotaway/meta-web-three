package com.metawebthree.notification.infrastructure.client;

import java.util.Optional;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.GetUserPhoneRequest;
import com.metawebthree.common.generated.rpc.GetUserPhoneResponse;
import com.metawebthree.common.generated.rpc.UserService;
import com.metawebthree.notification.domain.ports.UserQueryPort;

import lombok.extern.slf4j.Slf4j;

/**
 * User query port implementation using Dubbo RPC
 */
@Component
@Slf4j
public class UserQueryPortImpl implements UserQueryPort {

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    @Override
    public Optional<String> findNickname(Long userId) {
        log.debug("findNickname not implemented via RPC, userId: {}", userId);
        return Optional.empty();
    }

    @Override
    public Optional<String> findAvatar(Long userId) {
        log.debug("findAvatar not implemented via RPC, userId: {}", userId);
        return Optional.empty();
    }

    @Override
    public Optional<String> findPhone(Long userId) {
        try {
            log.info("Querying phone number for userId: {} via UserService RPC", userId);
            GetUserPhoneRequest request = GetUserPhoneRequest.newBuilder().setUserId(userId).build();
            GetUserPhoneResponse response = userService.getUserPhone(request);
            
            if (response.getSuccess() && response.getPhone() != null && !response.getPhone().isEmpty()) {
                log.info("Phone number retrieved successfully for userId: {}", userId);
                return Optional.of(response.getPhone());
            } else {
                log.warn("Phone number not found for userId: {}, message: {}", userId, response.getMessage());
                return Optional.empty();
            }
        } catch (Exception e) {
            log.error("Failed to query phone number for userId: {}, error: {}", userId, e.getMessage(), e);
            return Optional.empty();
        }
    }
}