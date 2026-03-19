package com.metawebthree.user.application.dto;

import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.domain.model.UserDO;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserDTO {
    private Long id;
    private String username;
    private String nickname;
    private String avatar;
    private String email;
    private String password;
    private UserRole userRoleId;
    private String walletAddress;
}
