package com.metawebthree.developerportal.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class OAuthAppCreateRequest {

    @NotBlank(message = "Application name is required")
    @Size(max = 128, message = "Name must not exceed 128 characters")
    private String name;

    @Size(max = 1000, message = "Description must not exceed 1000 characters")
    private String description;

    @NotBlank(message = "Redirect URIs are required")
    private String redirectUris;

    private String appType = "CONFIDENTIAL";

    private String grantTypes = "authorization_code,refresh_token";

    private String scopes;
}
