package com.metawebthree.developerportal.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * API Usage Statistics Entity
 * Records API call statistics for billing and monitoring
 */
@Data
@Entity
@Table(name = "api_usage_stats", indexes = {
    @Index(name = "idx_developer_time", columnList = "developer_id, stat_time"),
    @Index(name = "idx_api_endpoint", columnList = "api_endpoint, stat_time")
})
public class ApiUsageStats {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /**
     * Developer ID
     */
    @Column(name = "developer_id", nullable = false, length = 64)
    private String developerId;

    /**
     * API Key ID
     */
    @Column(name = "key_id", length = 64)
    private String keyId;

    /**
     * API endpoint called
     */
    @Column(name = "api_endpoint", nullable = false, length = 256)
    private String apiEndpoint;

    /**
     * HTTP method
     */
    @Column(name = "http_method", nullable = false, length = 16)
    private String httpMethod;

    /**
     * Statistics timestamp (hourly granularity)
     */
    @Column(name = "stat_time", nullable = false)
    private LocalDateTime statTime;

    /**
     * Total request count
     */
    @Column(name = "request_count", nullable = false)
    private Long requestCount = 0L;

    /**
     * Success count (HTTP 2xx)
     */
    @Column(name = "success_count", nullable = false)
    private Long successCount = 0L;

    /**
     * Error count (HTTP 4xx/5xx)
     */
    @Column(name = "error_count", nullable = false)
    private Long errorCount = 0L;

    /**
     * Average response time in milliseconds
     */
    @Column(name = "avg_response_time_ms")
    private Double avgResponseTimeMs;

    /**
     * Total data transferred in bytes
     */
    @Column(name = "data_transferred_bytes")
    private Long dataTransferredBytes = 0L;

    /**
     * Total billing amount in cents
     */
    @Column(name = "billing_amount_cents", nullable = false)
    private Long billingAmountCents = 0L;

    /**
     * Created timestamp
     */
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @PrePersist
    public void prePersist() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }
}
