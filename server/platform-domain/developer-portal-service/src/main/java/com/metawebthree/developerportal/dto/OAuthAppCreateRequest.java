package com.metawebthree.developerportal.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * OAuth Application Creation Request DTO
 */
@Data
public class OAuthAppCreateRequest {

    @NotBlank(message = "Application name is required")
    @Size(max = 128, message = "Name must not exceed 128 characters")
    private String name;

    @Size(max = 1000, message = "Description must not exceed 1000 characters")
    private String description;

    /**
     * Redirect URIs (comma-separated)
     */
    @NotBlank(message = "Redirect URIs are required")
    private String redirectUris;

    /**
     * Application type: CONFIDENTIAL or PUBLIC
     */
    private String appType = "CONFIDENTIAL";

    /**
     * Grant types (comma-separated)
     */
    private String grantTypes = "authorization_code,refresh_token";

    /**
     * Allowed scopes (comma-separated)
     */
    private String scopes;
}
