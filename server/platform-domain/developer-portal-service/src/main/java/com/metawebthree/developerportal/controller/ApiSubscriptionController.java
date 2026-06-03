package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.service.ApiSubscriptionService;
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
 * API Subscription Controller
 * Handles API subscription approval and management
 */
@Tag(name = "API Subscriptions", description = "API subscription management")
@RestController
@RequestMapping("/developer/subscriptions")
@RequiredArgsConstructor
public class ApiSubscriptionController {

    private final ApiSubscriptionService subscriptionService;

    // ==================== Developer: Subscription Management ====================

    @Operation(summary = "Request API subscription", description = "Submit a request to subscribe to specific API endpoints")
    @PostMapping("/{developerId}")
    public ResponseEntity<ApiSubscriptionResponse> requestSubscription(
        @PathVariable String developerId,
        @Valid @RequestBody ApiSubscriptionRequest request
    ) {
        ApiSubscriptionResponse response = subscriptionService.requestSubscription(developerId, request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @Operation(summary = "Get subscription", description = "Get subscription details by ID")
    @GetMapping("/{subscriptionId}")
    public ResponseEntity<ApiSubscriptionResponse> getSubscription(@PathVariable String subscriptionId) {
        ApiSubscriptionResponse response = subscriptionService.getSubscription(subscriptionId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "List developer subscriptions", description = "Get all subscriptions for a developer")
    @GetMapping("/developer/{developerId}")
    public ResponseEntity<List<ApiSubscriptionResponse>> getDeveloperSubscriptions(@PathVariable String developerId) {
        List<ApiSubscriptionResponse> response = subscriptionService.getDeveloperSubscriptions(developerId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Cancel subscription", description = "Cancel an active subscription")
    @PostMapping("/{subscriptionId}/cancel")
    public ResponseEntity<ApiSubscriptionResponse> cancelSubscription(@PathVariable String subscriptionId) {
        ApiSubscriptionResponse response = subscriptionService.cancelSubscription(subscriptionId);
        return ResponseEntity.ok(response);
    }

    // ==================== Admin: Approval Management ====================

    @Operation(summary = "Get pending subscriptions", description = "List all subscriptions awaiting approval (Admin only)")
    @GetMapping("/admin/pending")
    public ResponseEntity<List<ApiSubscriptionResponse>> getPendingSubscriptions() {
        List<ApiSubscriptionResponse> response = subscriptionService.getPendingSubscriptions();
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Get active subscriptions", description = "List all active subscriptions (Admin only)")
    @GetMapping("/admin/active")
    public ResponseEntity<List<ApiSubscriptionResponse>> getActiveSubscriptions() {
        List<ApiSubscriptionResponse> response = subscriptionService.getActiveSubscriptions();
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Approve subscription", description = "Approve a pending subscription request (Admin only)")
    @PostMapping("/admin/{subscriptionId}/approve")
    public ResponseEntity<ApiSubscriptionResponse> approveSubscription(
        @PathVariable String subscriptionId,
        @RequestBody Map<String, String> body
    ) {
        String reviewedBy = body.getOrDefault("reviewedBy", "admin");
        String note = body.get("note");
        ApiSubscriptionResponse response = subscriptionService.approveSubscription(subscriptionId, reviewedBy, note);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Activate subscription", description = "Activate an approved subscription (Admin only)")
    @PostMapping("/admin/{subscriptionId}/activate")
    public ResponseEntity<ApiSubscriptionResponse> activateSubscription(@PathVariable String subscriptionId) {
        ApiSubscriptionResponse response = subscriptionService.activateSubscription(subscriptionId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Reject subscription", description = "Reject a pending subscription request (Admin only)")
    @PostMapping("/admin/{subscriptionId}/reject")
    public ResponseEntity<ApiSubscriptionResponse> rejectSubscription(
        @PathVariable String subscriptionId,
        @RequestBody Map<String, String> body
    ) {
        String reviewedBy = body.getOrDefault("reviewedBy", "admin");
        String reason = body.getOrDefault("reason", "Subscription rejected");
        ApiSubscriptionResponse response = subscriptionService.rejectSubscription(subscriptionId, reviewedBy, reason);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Suspend subscription", description = "Suspend an active subscription (Admin only)")
    @PostMapping("/admin/{subscriptionId}/suspend")
    public ResponseEntity<ApiSubscriptionResponse> suspendSubscription(
        @PathVariable String subscriptionId,
        @RequestBody Map<String, String> body
    ) {
        String reason = body.getOrDefault("reason", "Subscription suspended");
        ApiSubscriptionResponse response = subscriptionService.suspendSubscription(subscriptionId, reason);
        return ResponseEntity.ok(response);
    }
}
