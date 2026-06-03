package com.metawebthree.developerportal.dto;

import com.metawebthree.developerportal.entity.ApiSubscription;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * API Subscription Response DTO
 */
@Data
public class ApiSubscriptionResponse {

    private String subscriptionId;
    private String developerId;
    private String apiPattern;
    private ApiSubscription.SubscriptionStatus status;
    private String reviewNote;
    private String reviewedBy;
    private LocalDateTime reviewedAt;
    private LocalDateTime startedAt;
    private LocalDateTime endedAt;
    private String reason;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    /**
     * Auto-generated API Key ID upon activation (only set when first activation)
     */
    private String generatedKeyId;
    
    /**
     * Auto-generated API Key Secret (only shown once)
     */
    private String generatedKeySecret;

    /**
     * Convert entity to DTO
     */
    public static ApiSubscriptionResponse fromEntity(ApiSubscription subscription) {
        ApiSubscriptionResponse response = new ApiSubscriptionResponse();
        response.setSubscriptionId(subscription.getSubscriptionId());
        response.setDeveloperId(subscription.getDeveloperId());
        response.setApiPattern(subscription.getApiPattern());
        response.setStatus(subscription.getStatus());
        response.setReviewNote(subscription.getReviewNote());
        response.setReviewedBy(subscription.getReviewedBy());
        response.setReviewedAt(subscription.getReviewedAt());
        response.setStartedAt(subscription.getStartedAt());
        response.setEndedAt(subscription.getEndedAt());
        response.setReason(subscription.getReason());
        response.setCreatedAt(subscription.getCreatedAt());
        response.setUpdatedAt(subscription.getUpdatedAt());
        return response;
    }
}
