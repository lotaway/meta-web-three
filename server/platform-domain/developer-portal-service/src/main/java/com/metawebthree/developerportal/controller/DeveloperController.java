package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.service.ApiDeveloperService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Developer Portal Controller
 * Handles developer registration and management
 */
@Tag(name = "Developer Portal", description = "API for third-party developer management")
@RestController
@RequestMapping("/developer")
@RequiredArgsConstructor
public class DeveloperController {

    private final ApiDeveloperService developerService;

    // ==================== Developer Registration ====================

    @Operation(summary = "Register as a new developer", description = "Submit registration for API access")
    @PostMapping("/register")
    public ResponseEntity<DeveloperResponse> register(@Valid @RequestBody DeveloperRegistrationRequest request) {
        DeveloperResponse response = developerService.register(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @Operation(summary = "Get developer profile", description = "Get current developer profile by ID")
    @GetMapping("/{developerId}")
    public ResponseEntity<DeveloperResponse> getDeveloper(@PathVariable String developerId) {
        DeveloperResponse response = developerService.getDeveloper(developerId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get developer by email", description = "Find developer by email address")
    @GetMapping("/by-email/{email}")
    public ResponseEntity<DeveloperResponse> getDeveloperByEmail(@PathVariable String email) {
        DeveloperResponse response = developerService.getDeveloperByEmail(email);
        return ResponseEntity.ok(response);
    }

    // ==================== Admin: Approval Management ====================

    @Operation(summary = "Get pending developers", description = "List all developers awaiting approval (Admin only)")
    @GetMapping("/admin/pending")
    public ResponseEntity<List<DeveloperResponse>> getPendingDevelopers() {
        List<DeveloperResponse> response = developerService.getPendingDevelopers();
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get approved developers", description = "List all approved developers (Admin only)")
    @GetMapping("/admin/approved")
    public ResponseEntity<List<DeveloperResponse>> getApprovedDevelopers() {
        List<DeveloperResponse> response = developerService.getApprovedDevelopers();
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Approve developer", description = "Approve a pending developer registration (Admin only)")
    @PostMapping("/admin/{developerId}/approve")
    public ResponseEntity<DeveloperResponse> approveDeveloper(
        @PathVariable String developerId,
        @RequestBody Map<String, String> body
    ) {
        String reviewedBy = body.getOrDefault("reviewedBy", "admin");
        String note = body.get("note");
        DeveloperResponse response = developerService.approve(developerId, reviewedBy, note);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Reject developer", description = "Reject a pending developer registration (Admin only)")
    @PostMapping("/admin/{developerId}/reject")
    public ResponseEntity<DeveloperResponse> rejectDeveloper(
        @PathVariable String developerId,
        @RequestBody Map<String, String> body
    ) {
        String reviewedBy = body.getOrDefault("reviewedBy", "admin");
        String reason = body.getOrDefault("reason", "Registration rejected");
        DeveloperResponse response = developerService.reject(developerId, reviewedBy, reason);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Suspend developer", description = "Suspend an approved developer (Admin only)")
    @PostMapping("/admin/{developerId}/suspend")
    public ResponseEntity<DeveloperResponse> suspendDeveloper(
        @PathVariable String developerId,
        @RequestBody Map<String, String> body
    ) {
        String reason = body.getOrDefault("reason", "Account suspended");
        DeveloperResponse response = developerService.suspend(developerId, reason);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Reactivate developer", description = "Reactivate a suspended developer (Admin only)")
    @PostMapping("/admin/{developerId}/reactivate")
    public ResponseEntity<DeveloperResponse> reactivateDeveloper(@PathVariable String developerId) {
        DeveloperResponse response = developerService.reactivate(developerId);
        return ResponseEntity.ok(response);
    }

    // ==================== Billing Plan Management ====================

    @Operation(summary = "Update billing plan", description = "Change developer's billing plan")
    @PostMapping("/{developerId}/billing-plan")
    public ResponseEntity<DeveloperResponse> updateBillingPlan(
        @PathVariable String developerId,
        @RequestBody Map<String, String> body
    ) {
        String planStr = body.get("plan");
        if (planStr == null) {
            return ResponseEntity.badRequest().build();
        }
        
        com.metawebthree.developerportal.entity.ApiDeveloper.BillingPlan plan;
        try {
            plan = com.metawebthree.developerportal.entity.ApiDeveloper.BillingPlan.valueOf(planStr.toUpperCase());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }
        
        DeveloperResponse response = developerService.updateBillingPlan(developerId, plan);
        return ResponseEntity.ok(response);
    }
}
