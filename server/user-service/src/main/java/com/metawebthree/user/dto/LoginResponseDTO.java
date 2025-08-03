package com.metawebthree.user.DTO;

import com.metawebthree.user.DO.UserDO;
import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class LoginResponseDTO {
    private String token;
    private UserDO user;
    private String walletAddress;
    private String loginType;
}