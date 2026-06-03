package com.metawebthree.developerportal.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * API Key Entity
 * Represents an API key issued to a developer for API access
 */
@Data
@Entity
@Table(name = "api_key")
public class ApiKey {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /**
     * API Key ID (public identifier)
     */
    @Column(name = "key_id", unique = true, nullable = false, length = 64)
    private String keyId;

    /**
     * API Key secret (hashed)
     */
    @Column(name = "key_secret", nullable = false, length = 128)
    private String keySecret;

    /**
     * Associated developer ID
     */
    @Column(name = "developer_id", nullable = false, length = 64)
    private String developerId;

    /**
     * Key name/description
     */
    @Column(name = "name", nullable = false, length = 128)
    private String name;

    /**
     * Key status: ACTIVE, DISABLED, EXPIRED, REVOKED
     */
    @Column(name = "status", nullable = false, length = 32)
    @Enumerated(EnumType.STRING)
    private KeyStatus status = KeyStatus.ACTIVE;

    /**
     * Expiration time (null means never expires)
     */
    @Column(name = "expires_at")
    private LocalDateTime expiresAt;

    /**
     * Allowed scopes/permissions
     */
    @Column(name = "scopes", columnDefinition = "TEXT")
    private String scopes;

    /**
     * Allowed IP addresses (comma-separated, empty means all IPs)
     */
    @Column(name = "allowed_ips", columnDefinition = "TEXT")
    private String allowedIps;

    /**
     * Allowed domains for CORS (comma-separated)
     */
    @Column(name = "allowed_domains", columnDefinition = "TEXT")
    private String allowedDomains;

    /**
     * Rate limit per minute for this key
     */
    @Column(name = "rate_limit")
    private Integer rateLimit = 100;

    /**
     * Last used timestamp
     */
    @Column(name = "last_used_at")
    private LocalDateTime lastUsedAt;

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

    /**
     * Key status enumeration
     */
    public enum KeyStatus {
        ACTIVE,     // Key is active and can be used
        DISABLED,   // Temporarily disabled
        EXPIRED,    // Key has expired
        REVOKED     // Key has been permanently revoked
    }
}
