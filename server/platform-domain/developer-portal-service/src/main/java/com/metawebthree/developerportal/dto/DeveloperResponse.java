package com.metawebthree.developerportal.dto;

import com.metawebthree.developerportal.entity.ApiDeveloper;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * Developer Response DTO
 */
@Data
public class DeveloperResponse {

    private String developerId;
    private String email;
    private String name;
    private String phone;
    private String description;
    private ApiDeveloper.DeveloperStatus status;
    private String reviewNote;
    private String reviewedBy;
    private LocalDateTime reviewedAt;
    private Integer dailyQuota;
    private Integer monthlyQuota;
    private ApiDeveloper.BillingPlan billingPlan;
    private Long balance;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    /**
     * Convert entity to DTO
     */
    public static DeveloperResponse fromEntity(ApiDeveloper developer) {
        DeveloperResponse response = new DeveloperResponse();
        response.setDeveloperId(developer.getDeveloperId());
        response.setEmail(developer.getEmail());
        response.setName(developer.getName());
        response.setPhone(developer.getPhone());
        response.setDescription(developer.getDescription());
        response.setStatus(developer.getStatus());
        response.setReviewNote(developer.getReviewNote());
        response.setReviewedBy(developer.getReviewedBy());
        response.setReviewedAt(developer.getReviewedAt());
        response.setDailyQuota(developer.getDailyQuota());
        response.setMonthlyQuota(developer.getMonthlyQuota());
        response.setBillingPlan(developer.getBillingPlan());
        response.setBalance(developer.getBalance());
        response.setCreatedAt(developer.getCreatedAt());
        response.setUpdatedAt(developer.getUpdatedAt());
        return response;
    }
}
