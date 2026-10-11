package com.metawebthree.crm.adapter.vo;

public record UserInfoDTO(Long id, String username, String phone, String email, String avatar, Integer status, Long createdAt) {
}