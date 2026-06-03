package com.metawebthree.developerportal.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * Developer Registration Request DTO
 */
@Data
public class DeveloperRegistrationRequest {

    @NotBlank(message = "Email is required")
    @Email(message = "Invalid email format")
    private String email;

    @NotBlank(message = "Name is required")
    @Size(max = 128, message = "Name must not exceed 128 characters")
    private String name;

    @Size(max = 32, message = "Phone must not exceed 32 characters")
    private String phone;

    @Size(max = 1000, message = "Description must not exceed 1000 characters")
    private String description;
}
