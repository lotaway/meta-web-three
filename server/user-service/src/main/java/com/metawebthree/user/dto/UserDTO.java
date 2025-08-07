package com.metawebthree.user.DTO;

import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.DO.UserDO;

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
