package com.metawebthree.developerportal.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * API Subscription Entity
 * Represents a developer's subscription to specific API endpoints
 */
@Data
@Entity
@Table(name = "api_subscription")
public class ApiSubscription {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /**
     * Subscription ID
     */
    @Column(name = "subscription_id", unique = true, nullable = false, length = 64)
    private String subscriptionId;

    /**
     * Developer ID
     */
    @Column(name = "developer_id", nullable = false, length = 64)
    private String developerId;

    /**
     * API endpoint pattern (e.g., /order-service/**, /product-service/api/v1/**)
     */
    @Column(name = "api_pattern", nullable = false, length = 256)
    private String apiPattern;

    /**
     * Subscription status: PENDING, APPROVED, ACTIVE, SUSPENDED, CANCELLED
     */
    @Column(name = "status", nullable = false, length = 32)
    @Enumerated(EnumType.STRING)
    private SubscriptionStatus status = SubscriptionStatus.PENDING;

    /**
     * Approval/rejection note
     */
    @Column(name = "review_note", columnDefinition = "TEXT")
    private String reviewNote;

    /**
     * Approved by admin
     */
    @Column(name = "reviewed_by", length = 64)
    private String reviewedBy;

    /**
     * Approval timestamp
     */
    @Column(name = "reviewed_at")
    private LocalDateTime reviewedAt;

    /**
     * Subscription start time
     */
    @Column(name = "started_at")
    private LocalDateTime startedAt;

    /**
     * Subscription end time (null means ongoing)
     */
    @Column(name = "ended_at")
    private LocalDateTime endedAt;

    /**
     * Reason for subscription (provided by developer)
     */
    @Column(name = "reason", columnDefinition = "TEXT")
    private String reason;

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
     * Subscription status enumeration
     */
    public enum SubscriptionStatus {
        PENDING,    // Awaiting approval
        APPROVED,   // Approved but not yet active
        ACTIVE,     // Active and can be used
        SUSPENDED,  // Temporarily suspended
        CANCELLED   // Cancelled by developer or admin
    }
}
