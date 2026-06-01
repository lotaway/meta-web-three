package com.metawebthree.socialcommerce.domain.ports;

public interface UserPort {
    Boolean isValidUser(Long userId);
    String getUserNickname(Long userId);
}