package com.metawebthree.developerportal.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * API Developer Registration Entity
 * Represents a third-party developer who wants to access open APIs
 */
@Data
@Entity
@Table(name = "api_developer")
public class ApiDeveloper {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /**
     * Developer unique identifier
     */
    @Column(name = "developer_id", unique = true, nullable = false, length = 64)
    private String developerId;

    /**
     * Developer email for contact
     */
    @Column(name = "email", unique = true, nullable = false, length = 128)
    private String email;

    /**
     * Developer name / Company name
     */
    @Column(name = "name", nullable = false, length = 128)
    private String name;

    /**
     * Contact phone number
     */
    @Column(name = "phone", length = 32)
    private String phone;

    /**
     * Company description
     */
    @Column(name = "description", columnDefinition = "TEXT")
    private String description;

    /**
     * Developer status: PENDING, APPROVED, SUSPENDED, REJECTED
     */
    @Column(name = "status", nullable = false, length = 32)
    @Enumerated(EnumType.STRING)
    private DeveloperStatus status = DeveloperStatus.PENDING;

    /**
     * Approval/rejection reason
     */
    @Column(name = "review_note", columnDefinition = "TEXT")
    private String reviewNote;

    /**
     * Approved by admin
     */
    @Column(name = "reviewed_by", length = 64)
    private String reviewedBy;

    /**
     * Approval/rejection timestamp
     */
    @Column(name = "reviewed_at")
    private LocalDateTime reviewedAt;

    /**
     * Total API call quota per day
     */
    @Column(name = "daily_quota", nullable = false)
    private Integer dailyQuota = 10000;

    /**
     * Monthly quota
     */
    @Column(name = "monthly_quota", nullable = false)
    private Integer monthlyQuota = 100000;

    /**
     * Current billing plan: FREE, BASIC, PROFESSIONAL, ENTERPRISE
     */
    @Column(name = "billing_plan", nullable = false, length = 32)
    @Enumerated(EnumType.STRING)
    private BillingPlan billingPlan = BillingPlan.FREE;

    /**
     * Account balance (in cents)
     */
    @Column(name = "balance", nullable = false)
    private Long balance = 0L;

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
     * Developer status enumeration
     */
    public enum DeveloperStatus {
        PENDING,      // Registration pending approval
        APPROVED,     // Approved, can use APIs
        SUSPENDED,    // Suspended due to violation or payment issues
        REJECTED      // Registration rejected
    }

    /**
     * Billing plan enumeration
     */
    public enum BillingPlan {
        FREE(0, 10000, 100000),           // Free tier
        BASIC(2900, 50000, 500000),       // $29/month
        PROFESSIONAL(9900, 200000, 2000000), // $99/month
        ENTERPRISE(0, Integer.MAX_VALUE, Integer.MAX_VALUE); // Custom pricing

        private final int monthlyFeeCents;
        private final int dailyQuota;
        private final int monthlyQuota;

        BillingPlan(int monthlyFeeCents, int dailyQuota, int monthlyQuota) {
            this.monthlyFeeCents = monthlyFeeCents;
            this.dailyQuota = dailyQuota;
            this.monthlyQuota = monthlyQuota;
        }

        public int getMonthlyFeeCents() {
            return monthlyFeeCents;
        }

        public int getDailyQuota() {
            return dailyQuota;
        }

        public int getMonthlyQuota() {
            return monthlyQuota;
        }
    }
}
