package com.metawebthree.user.application.dto;
import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;

import java.util.Date;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class LoginResponseDTO {
    private String token;
    private String refreshToken;
    private Date refreshTokenExpiresAt;
    private UserDTO user;
    private String walletAddress;
    private String loginType;
}