package com.metawebthree.cs.infrastructure.client;

import java.util.Optional;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.cs.domain.ports.UserQueryPort;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class UserQueryPortImpl implements UserQueryPort {

    @DubboReference(check = false, lazy = true)
    private UserService userService;

    @Override
    public Optional<String> findNickname(Long userId) {
        // UserService does not provide nickname query via RPC
        // Could be extended in future when user-service adds more endpoints
        log.debug("findNickname not implemented via RPC, userId: {}", userId);
        return Optional.empty();
    }

    @Override
    public Optional<String> findAvatar(Long userId) {
        // UserService does not provide avatar query via RPC
        // Could be extended in future when user-service adds more endpoints
        log.debug("findAvatar not implemented via RPC, userId: {}", userId);
        return Optional.empty();
    }

    @Override
    public Optional<String> findPhone(Long userId) {
        // UserService does not provide phone query via RPC
        // Could be extended in future when user-service adds more endpoints
        log.debug("findPhone not implemented via RPC, userId: {}", userId);
        return Optional.empty();
    }
}