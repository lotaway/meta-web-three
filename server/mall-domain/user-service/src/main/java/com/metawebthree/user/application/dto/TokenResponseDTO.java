package com.metawebthree.user.application.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Date;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TokenResponseDTO {
    private String accessToken;
    private String refreshToken;
    private Date accessTokenExpiresAt;
    private Date refreshTokenExpiresAt;
    private Long expiresInSeconds;
}