package com.metawebthree.tenant.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class RegisterRequest {
    @NotBlank(message = "Tenant name is required")
    private String name;

    @NotBlank(message = "Tenant code is required")
    private String code;

    @NotBlank(message = "Contact email is required")
    @Email(message = "Invalid email format")
    private String contactEmail;

    private String contactName;
    private String contactPhone;

    @NotBlank(message = "CAPTCHA token is required")
    private String captchaToken;

    @NotBlank(message = "CAPTCHA answer is required")
    private String captchaAnswer;

    @NotBlank(message = "Email verification code is required")
    private String emailCode;
}
