package com.metawebthree.user.application.dto;

import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SsoLoginResponseDTO {
    private String token;
    private String tokenHead;
}
