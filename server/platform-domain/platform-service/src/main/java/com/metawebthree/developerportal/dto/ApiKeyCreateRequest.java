package com.metawebthree.developerportal.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ApiKeyCreateRequest {

    @NotBlank(message = "Key name is required")
    @Size(max = 128, message = "Name must not exceed 128 characters")
    private String name;

    private LocalDateTime expiresAt;

    private String scopes;

    private String allowedIps;

    private String allowedDomains;

    private Integer rateLimit = 100;
}
