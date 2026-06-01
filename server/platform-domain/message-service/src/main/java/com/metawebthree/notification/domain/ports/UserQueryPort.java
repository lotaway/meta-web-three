package com.metawebthree.notification.domain.ports;

import java.util.Optional;

/**
 * User query port for retrieving user information
 */
public interface UserQueryPort {
    Optional<String> findNickname(Long userId);
    Optional<String> findAvatar(Long userId);
    Optional<String> findPhone(Long userId);
}