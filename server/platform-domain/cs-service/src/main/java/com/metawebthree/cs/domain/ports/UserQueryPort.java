package com.metawebthree.cs.domain.ports;

import java.util.Optional;

public interface UserQueryPort {
    Optional<String> findNickname(Long userId);
    Optional<String> findAvatar(Long userId);
    Optional<String> findPhone(Long userId);
}
