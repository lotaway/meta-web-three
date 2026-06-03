package com.metawebthree.developerportal.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * API Key Creation Request DTO
 */
@Data
public class ApiKeyCreateRequest {

    @NotBlank(message = "Key name is required")
    @Size(max = 128, message = "Name must not exceed 128 characters")
    private String name;

    /**
     * Expiration time (optional, null means never expires)
     */
    private LocalDateTime expiresAt;

    /**
     * Allowed scopes (comma-separated)
     */
    private String scopes;

    /**
     * Allowed IP addresses (comma-separated)
     */
    private String allowedIps;

    /**
     * Allowed domains for CORS (comma-separated)
     */
    private String allowedDomains;

    /**
     * Rate limit per minute (default 100)
     */
    private Integer rateLimit = 100;
}
