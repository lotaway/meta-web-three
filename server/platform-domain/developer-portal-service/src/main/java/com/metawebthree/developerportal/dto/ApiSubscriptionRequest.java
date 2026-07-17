package com.metawebthree.developerportal.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class ApiSubscriptionRequest {

    @NotBlank(message = "API pattern is required")
    @Size(max = 256, message = "API pattern must not exceed 256 characters")
    private String apiPattern;

    @Size(max = 1000, message = "Reason must not exceed 1000 characters")
    private String reason;
}
