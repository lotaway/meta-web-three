package com.metawebthree.developerportal.dto;

import com.metawebthree.developerportal.entity.ApiKey;
import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ApiKeyResponse {

    private Long id;
    private String keyId;
    private String keySecret;
    private String developerId;
    private String name;
    private ApiKey.KeyStatus status;
    private LocalDateTime expiresAt;
    private String scopes;
    private String allowedIps;
    private String allowedDomains;
    private Integer rateLimit;
    private LocalDateTime lastUsedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static ApiKeyResponse fromEntity(ApiKey apiKey) {
        ApiKeyResponse response = new ApiKeyResponse();
        response.setId(apiKey.getId());
        response.setKeyId(apiKey.getKeyId());
        response.setDeveloperId(apiKey.getDeveloperId());
        response.setName(apiKey.getName());
        response.setStatus(apiKey.getStatus());
        response.setExpiresAt(apiKey.getExpiresAt());
        response.setScopes(apiKey.getScopes());
        response.setAllowedIps(apiKey.getAllowedIps());
        response.setAllowedDomains(apiKey.getAllowedDomains());
        response.setRateLimit(apiKey.getRateLimit());
        response.setLastUsedAt(apiKey.getLastUsedAt());
        response.setCreatedAt(apiKey.getCreatedAt());
        response.setUpdatedAt(apiKey.getUpdatedAt());
        return response;
    }
}
