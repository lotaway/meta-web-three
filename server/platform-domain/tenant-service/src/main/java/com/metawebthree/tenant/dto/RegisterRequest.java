package com.metawebthree.tenant.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class RegisterRequest {
    @NotBlank(message = "Tenant name is required")
    @Size(max = 50, message = "Tenant name must be at most 50 characters")
    private String name;

    @NotBlank(message = "Tenant code is required")
    @Size(min = 4, max = 20, message = "Tenant code must be between 4 and 20 characters")
    private String code;

    @NotBlank(message = "Contact email is required")
    @Email(message = "Invalid email format")
    @Size(max = 100, message = "Contact email must be at most 100 characters")
    private String contactEmail;

    @Size(max = 50, message = "Contact name must be at most 50 characters")
    private String contactName;

    @Size(max = 20, message = "Contact phone must be at most 20 characters")
    private String contactPhone;

    @NotBlank(message = "CAPTCHA token is required")
    @Size(max = 200, message = "Invalid CAPTCHA token")
    private String captchaToken;

    @NotBlank(message = "CAPTCHA answer is required")
    @Size(max = 8, message = "CAPTCHA answer must be at most 8 characters")
    private String captchaAnswer;

    @NotBlank(message = "Email verification code is required")
    @Size(min = 6, max = 6, message = "Email verification code must be exactly 6 digits")
    private String emailCode;
}
