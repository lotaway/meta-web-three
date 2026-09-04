package com.metawebthree.developerportal.controller;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.registration.EmailVerificationCodeService;
import com.metawebthree.common.registration.IpRateLimitService;
import com.metawebthree.common.registration.TokenCaptchaService;
import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.service.ApiDeveloperService;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Tag(name = "Developer Portal", description = "API for third-party developer management")
@RestController
@RequestMapping("/developer")
@RequiredArgsConstructor
public class DeveloperController {

    private final ApiDeveloperService developerService;
    private final TokenCaptchaService developerCaptchaService;
    private final EmailVerificationCodeService developerEmailVerificationService;
    private final IpRateLimitService developerCaptchaRateLimiter;
    private final IpRateLimitService developerEmailRateLimiter;
    private final IpRateLimitService developerRegisterRateLimiter;

    @Operation(summary = "Generate registration CAPTCHA", description = "Generate an image CAPTCHA challenge for developer registration")
    @GetMapping("/captcha/generate")
    public ApiResponse<TokenCaptchaService.CaptchaChallenge> generateCaptcha(HttpServletRequest request) {
        if (!developerCaptchaRateLimiter.isAllowed(request)) {
            throw new BusinessException(ResponseStatus.REGISTRATION_RATE_LIMITED);
        }
        return ApiResponse.success(developerCaptchaService.generate());
    }

    @Operation(summary = "Send email verification code", description = "Send a verification code to the given email address")
    @PostMapping("/email/send-verification-code")
    public ApiResponse<Void> sendVerificationCode(HttpServletRequest request, @RequestBody Map<String, String> body) {
        if (!developerEmailRateLimiter.isAllowed(request)) {
            throw new BusinessException(ResponseStatus.REGISTRATION_RATE_LIMITED);
        }
        String email = body.get("email");
        if (email == null || email.isBlank()) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "Email is required");
        }
        if (developerService.existsByEmail(email)) {
            throw new BusinessException(ResponseStatus.DEVELOPER_ALREADY_EXISTS);
        }
        developerEmailVerificationService.sendCode(email);
        return ApiResponse.success();
    }

    @RateLimiter(name = "developerRegister")
    @Operation(summary = "Register as a new developer", description = "Submit registration for API access with CAPTCHA and email verification")
    @PostMapping("/register")
    public ApiResponse<DeveloperResponse> register(HttpServletRequest request,
                                                   @Valid @RequestBody DeveloperRegistrationRequest registerRequest) {
        if (!developerRegisterRateLimiter.isAllowed(request)) {
            throw new BusinessException(ResponseStatus.REGISTRATION_RATE_LIMITED);
        }
        if (!developerCaptchaService.verify(registerRequest.getCaptchaToken(), registerRequest.getCaptchaAnswer())) {
            throw new BusinessException(ResponseStatus.CAPTCHA_INVALID);
        }
        if (!developerEmailVerificationService.verifyCode(registerRequest.getEmail(), registerRequest.getEmailCode())) {
            throw new BusinessException(ResponseStatus.EMAIL_VERIFICATION_CODE_INVALID);
        }
        DeveloperResponse response = developerService.register(registerRequest);
        return ApiResponse.success(response);
    }

    @Operation(summary = "Get developer profile", description = "Get current developer profile by ID")
    @GetMapping("/{developerId}")
    public ApiResponse<DeveloperResponse> getDeveloper(@PathVariable String developerId) {
        return ApiResponse.success(developerService.getDeveloper(developerId));
    }

    @Operation(summary = "Get developer by email", description = "Find developer by email address")
    @GetMapping("/by-email/{email}")
    public ApiResponse<DeveloperResponse> getDeveloperByEmail(@PathVariable String email) {
        return ApiResponse.success(developerService.getDeveloperByEmail(email));
    }

    @Operation(summary = "Get pending developers", description = "List all developers awaiting approval (Admin only)")
    @GetMapping("/admin/pending")
    public ApiResponse<List<DeveloperResponse>> getPendingDevelopers(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole) {
        ensureAdmin(userRole);
        return ApiResponse.success(developerService.getPendingDevelopers());
    }

    @Operation(summary = "Get approved developers", description = "List all approved developers (Admin only)")
    @GetMapping("/admin/approved")
    public ApiResponse<List<DeveloperResponse>> getApprovedDevelopers(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole) {
        ensureAdmin(userRole);
        return ApiResponse.success(developerService.getApprovedDevelopers());
    }

    @Operation(summary = "Approve developer", description = "Approve a pending developer registration (Admin only)")
    @PostMapping("/admin/{developerId}/approve")
    public ApiResponse<DeveloperResponse> approveDeveloper(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole,
            @PathVariable String developerId,
            @RequestBody Map<String, String> body) {
        ensureAdmin(userRole);
        String reviewedBy = body.getOrDefault("reviewedBy", "admin");
        String note = body.get("note");
        return ApiResponse.success(developerService.approve(developerId, reviewedBy, note));
    }

    @Operation(summary = "Reject developer", description = "Reject a pending developer registration (Admin only)")
    @PostMapping("/admin/{developerId}/reject")
    public ApiResponse<DeveloperResponse> rejectDeveloper(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole,
            @PathVariable String developerId,
            @RequestBody Map<String, String> body) {
        ensureAdmin(userRole);
        String reviewedBy = body.getOrDefault("reviewedBy", "admin");
        String reason = body.getOrDefault("reason", "Registration rejected");
        return ApiResponse.success(developerService.reject(developerId, reviewedBy, reason));
    }

    @Operation(summary = "Suspend developer", description = "Suspend an approved developer (Admin only)")
    @PostMapping("/admin/{developerId}/suspend")
    public ApiResponse<DeveloperResponse> suspendDeveloper(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole,
            @PathVariable String developerId,
            @RequestBody Map<String, String> body) {
        ensureAdmin(userRole);
        String reason = body.getOrDefault("reason", "Account suspended");
        return ApiResponse.success(developerService.suspend(developerId, reason));
    }

    @Operation(summary = "Reactivate developer", description = "Reactivate a suspended developer (Admin only)")
    @PostMapping("/admin/{developerId}/reactivate")
    public ApiResponse<DeveloperResponse> reactivateDeveloper(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole,
            @PathVariable String developerId) {
        ensureAdmin(userRole);
        return ApiResponse.success(developerService.reactivate(developerId));
    }

    @Operation(summary = "Update billing plan", description = "Change developer's billing plan (Admin only)")
    @PostMapping("/{developerId}/billing-plan")
    public ApiResponse<DeveloperResponse> updateBillingPlan(
            @RequestHeader(value = HeaderConstants.USER_ROLE, required = false) String userRole,
            @PathVariable String developerId,
            @RequestBody Map<String, String> body) {
        ensureAdmin(userRole);
        String planStr = body.get("plan");
        if (planStr == null) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "Plan is required");
        }

        ApiDeveloper.BillingPlan plan;
        try {
            plan = ApiDeveloper.BillingPlan.valueOf(planStr.toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Invalid billing plan");
        }

        return ApiResponse.success(developerService.updateBillingPlan(developerId, plan));
    }

    private void ensureAdmin(String userRole) {
        if (!"ADMIN".equals(userRole)) {
            throw new BusinessException(ResponseStatus.FORBIDDEN);
        }
    }
}