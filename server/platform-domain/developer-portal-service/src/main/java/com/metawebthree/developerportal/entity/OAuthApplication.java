package com.metawebthree.developerportal.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * OAuth Application Entity
 * Represents an OAuth 2.0 application registered by a developer
 */
@Data
@Entity
@Table(name = "oauth_application")
public class OAuthApplication {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /**
     * OAuth Client ID
     */
    @Column(name = "client_id", unique = true, nullable = false, length = 64)
    private String clientId;

    /**
     * OAuth Client Secret (hashed)
     */
    @Column(name = "client_secret", nullable = false, length = 128)
    private String clientSecret;

    /**
     * Developer ID who owns this application
     */
    @Column(name = "developer_id", nullable = false, length = 64)
    private String developerId;

    /**
     * Application name
     */
    @Column(name = "name", nullable = false, length = 128)
    private String name;

    /**
     * Application description
     */
    @Column(name = "description", columnDefinition = "TEXT")
    private String description;

    /**
     * Redirect URIs (comma-separated)
     */
    @Column(name = "redirect_uris", columnDefinition = "TEXT")
    private String redirectUris;

    /**
     * Application type: CONFIDENTIAL, PUBLIC
     */
    @Column(name = "app_type", nullable = false, length = 32)
    @Enumerated(EnumType.STRING)
    private AppType appType = AppType.CONFIDENTIAL;

    /**
     * Grant types supported (comma-separated): authorization_code, client_credentials, refresh_token
     */
    @Column(name = "grant_types", columnDefinition = "TEXT")
    private String grantTypes = "authorization_code,refresh_token";

    /**
     * Allowed scopes
     */
    @Column(name = "scopes", columnDefinition = "TEXT")
    private String scopes;

    /**
     * Application status: ACTIVE, DISABLED, SUSPENDED
     */
    @Column(name = "status", nullable = false, length = 32)
    @Enumerated(EnumType.STRING)
    private AppStatus status = AppStatus.ACTIVE;

    /**
     * Created timestamp
     */
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    /**
     * Last updated timestamp
     */
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    public void prePersist() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
        if (updatedAt == null) {
            updatedAt = LocalDateTime.now();
        }
    }

    @PreUpdate
    public void preUpdate() {
        updatedAt = LocalDateTime.now();
    }

    public enum AppType {
        CONFIDENTIAL,  // Server-side application with client secret
        PUBLIC         // Mobile/SPA without client secret
    }

    public enum AppStatus {
        ACTIVE,
        DISABLED,
        SUSPENDED
    }
}
