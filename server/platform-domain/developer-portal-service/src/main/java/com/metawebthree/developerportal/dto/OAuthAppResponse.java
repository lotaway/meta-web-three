package com.metawebthree.developerportal.dto;

import com.metawebthree.developerportal.entity.OAuthApplication;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * OAuth Application Response DTO
 */
@Data
public class OAuthAppResponse {

    private Long id;
    private String clientId;
    private String clientSecret; // Only returned on creation
    private String developerId;
    private String name;
    private String description;
    private String redirectUris;
    private OAuthApplication.AppType appType;
    private String grantTypes;
    private String scopes;
    private OAuthApplication.AppStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    /**
     * Convert entity to DTO
     */
    public static OAuthAppResponse fromEntity(OAuthApplication app) {
        OAuthAppResponse response = new OAuthAppResponse();
        response.setId(app.getId());
        response.setClientId(app.getClientId());
        response.setDeveloperId(app.getDeveloperId());
        response.setName(app.getName());
        response.setDescription(app.getDescription());
        response.setRedirectUris(app.getRedirectUris());
        response.setAppType(app.getAppType());
        response.setGrantTypes(app.getGrantTypes());
        response.setScopes(app.getScopes());
        response.setStatus(app.getStatus());
        response.setCreatedAt(app.getCreatedAt());
        response.setUpdatedAt(app.getUpdatedAt());
        return response;
    }
}
