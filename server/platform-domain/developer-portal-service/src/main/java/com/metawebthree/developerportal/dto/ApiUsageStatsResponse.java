package com.metawebthree.developerportal.dto;

import lombok.Data;

/**
 * API Usage Statistics Response DTO
 */
@Data
public class ApiUsageStatsResponse {

    private String developerId;
    private String apiEndpoint;
    private String httpMethod;
    private Long totalRequests;
    private Long successCount;
    private Long errorCount;
    private Double avgResponseTimeMs;
    private Long dataTransferredBytes;
    private Long billingAmountCents;
    private Double errorRate;
}
