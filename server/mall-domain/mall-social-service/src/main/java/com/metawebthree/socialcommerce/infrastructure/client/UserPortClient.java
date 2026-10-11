package com.metawebthree.socialcommerce.infrastructure.client;

import org.springframework.stereotype.Component;

import com.metawebthree.socialcommerce.domain.ports.UserPort;

/**
 * 用户端口适配器（本地开发实现）。
 * 尚未接入 user-service 的真实 RPC，做最小可用校验。
 */
@Component
public class UserPortClient implements UserPort {

    @Override
    public Boolean isValidUser(Long userId) {
        return userId != null;
    }

    @Override
    public String getUserNickname(Long userId) {
        return "用户" + userId;
    }
}
