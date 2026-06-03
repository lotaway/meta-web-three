package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.entity.ApiKey;
import com.metawebthree.developerportal.entity.ApiSubscription;
import com.metawebthree.developerportal.entity.ApiSubscription.SubscriptionStatus;
import com.metawebthree.developerportal.repository.ApiDeveloperRepository;
import com.metawebthree.developerportal.repository.ApiKeyRepository;
import com.metawebthree.developerportal.repository.ApiSubscriptionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * API Subscription Service
 * Handles API subscription approval and management
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ApiSubscriptionService {

    private final ApiSubscriptionRepository subscriptionRepository;
    private final ApiDeveloperRepository developerRepository;
    private final ApiKeyRepository apiKeyRepository;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    /**
     * Request a new API subscription
     */
    @Transactional
    public ApiSubscriptionResponse requestSubscription(String developerId, ApiSubscriptionRequest request) {
        // Validate developer exists and is approved
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));
        
        if (developer.getStatus() != ApiDeveloper.DeveloperStatus.APPROVED) {
            throw new IllegalStateException("Developer is not approved: " + developer.getStatus());
        }

        // Check if subscription already exists
        if (subscriptionRepository.existsByDeveloperIdAndApiPattern(developerId, request.getApiPattern())) {
            throw new IllegalArgumentException("Subscription already exists for this API pattern");
        }

        ApiSubscription subscription = new ApiSubscription();
        subscription.setSubscriptionId(generateSubscriptionId());
        subscription.setDeveloperId(developerId);
        subscription.setApiPattern(request.getApiPattern());
        subscription.setReason(request.getReason());
        subscription.setStatus(SubscriptionStatus.PENDING);
        
        subscription = subscriptionRepository.save(subscription);
        log.info("API Subscription requested: {} for developer {}", request.getApiPattern(), developerId);
        
        return ApiSubscriptionResponse.fromEntity(subscription);
    }

    /**
     * Approve a pending subscription
     */
    @Transactional
    public ApiSubscriptionResponse approveSubscription(String subscriptionId, String reviewedBy, String note) {
        ApiSubscription subscription = subscriptionRepository.findBySubscriptionId(subscriptionId)
            .orElseThrow(() -> new IllegalArgumentException("Subscription not found: " + subscriptionId));
        
        if (subscription.getStatus() != SubscriptionStatus.PENDING) {
            throw new IllegalStateException("Subscription is not in PENDING status: " + subscription.getStatus());
        }
        
        subscription.setStatus(SubscriptionStatus.APPROVED);
        subscription.setReviewedBy(reviewedBy);
        subscription.setReviewNote(note);
        subscription.setReviewedAt(LocalDateTime.now());
        
        subscription = subscriptionRepository.save(subscription);
        log.info("API Subscription approved: {} by {}", subscriptionId, reviewedBy);
        
        return ApiSubscriptionResponse.fromEntity(subscription);
    }

    /**
     * Activate an approved subscription and auto-generate API Key
     */
    @Transactional
    public ApiSubscriptionResponse activateSubscription(String subscriptionId) {
        ApiSubscription subscription = subscriptionRepository.findBySubscriptionId(subscriptionId)
            .orElseThrow(() -> new IllegalArgumentException("Subscription not found: " + subscriptionId));
        
        if (subscription.getStatus() != SubscriptionStatus.APPROVED) {
            throw new IllegalStateException("Subscription must be APPROVED before activation");
        }
        
        subscription.setStatus(SubscriptionStatus.ACTIVE);
        subscription.setStartedAt(LocalDateTime.now());
        subscription = subscriptionRepository.save(subscription);
        
        // Auto-generate API Key for the developer if not exists
        String developerId = subscription.getDeveloperId();
        List<ApiKey> existingKeys = apiKeyRepository.findByDeveloperId(developerId);
        
        String generatedKeyId = null;
        String generatedKeySecret = null;
        if (existingKeys.stream().noneMatch(k -> k.getStatus() == ApiKey.KeyStatus.ACTIVE)) {
            // Generate a new API key
            generatedKeyId = generateKeyId();
            generatedKeySecret = generateKeySecret();
            
            ApiKey apiKey = new ApiKey();
            apiKey.setKeyId(generatedKeyId);
            apiKey.setKeySecret(passwordEncoder.encode(generatedKeySecret));
            apiKey.setDeveloperId(developerId);
            apiKey.setName("Auto-generated for subscription: " + subscriptionId);
            apiKey.setStatus(ApiKey.KeyStatus.ACTIVE);
            apiKey.setScopes(subscription.getApiPattern());
            apiKey.setRateLimit(100);
            
            apiKeyRepository.save(apiKey);
            log.info("Auto-generated API Key {} for developer {} upon subscription activation", 
                generatedKeyId, developerId);
        }
        
        log.info("API Subscription activated: {}", subscriptionId);
        
        ApiSubscriptionResponse response = ApiSubscriptionResponse.fromEntity(subscription);
        if (generatedKeyId != null) {
            response.setGeneratedKeyId(generatedKeyId);
            response.setGeneratedKeySecret(generatedKeySecret);
        }
        return response;
    }

    /**
     * Reject a pending subscription
     */
    @Transactional
    public ApiSubscriptionResponse rejectSubscription(String subscriptionId, String reviewedBy, String reason) {
        ApiSubscription subscription = subscriptionRepository.findBySubscriptionId(subscriptionId)
            .orElseThrow(() -> new IllegalArgumentException("Subscription not found: " + subscriptionId));
        
        if (subscription.getStatus() != SubscriptionStatus.PENDING) {
            throw new IllegalStateException("Subscription is not in PENDING status: " + subscription.getStatus());
        }
        
        subscription.setStatus(SubscriptionStatus.CANCELLED);
        subscription.setReviewedBy(reviewedBy);
        subscription.setReviewNote(reason);
        subscription.setReviewedAt(LocalDateTime.now());
        
        subscription = subscriptionRepository.save(subscription);
        log.info("API Subscription rejected: {} by {}", subscriptionId, reviewedBy);
        
        return ApiSubscriptionResponse.fromEntity(subscription);
    }

    /**
     * Suspend an active subscription
     */
    @Transactional
    public ApiSubscriptionResponse suspendSubscription(String subscriptionId, String reason) {
        ApiSubscription subscription = subscriptionRepository.findBySubscriptionId(subscriptionId)
            .orElseThrow(() -> new IllegalArgumentException("Subscription not found: " + subscriptionId));
        
        subscription.setStatus(SubscriptionStatus.SUSPENDED);
        subscription.setReviewNote(reason);
        
        subscription = subscriptionRepository.save(subscription);
        log.info("API Subscription suspended: {} - {}", subscriptionId, reason);
        
        return ApiSubscriptionResponse.fromEntity(subscription);
    }

    /**
     * Cancel a subscription
     */
    @Transactional
    public ApiSubscriptionResponse cancelSubscription(String subscriptionId) {
        ApiSubscription subscription = subscriptionRepository.findBySubscriptionId(subscriptionId)
            .orElseThrow(() -> new IllegalArgumentException("Subscription not found: " + subscriptionId));
        
        subscription.setStatus(SubscriptionStatus.CANCELLED);
        subscription.setEndedAt(LocalDateTime.now());
        
        subscription = subscriptionRepository.save(subscription);
        log.info("API Subscription cancelled: {}", subscriptionId);
        
        return ApiSubscriptionResponse.fromEntity(subscription);
    }

    /**
     * Get subscription by ID
     */
    public ApiSubscriptionResponse getSubscription(String subscriptionId) {
        ApiSubscription subscription = subscriptionRepository.findBySubscriptionId(subscriptionId)
            .orElseThrow(() -> new IllegalArgumentException("Subscription not found: " + subscriptionId));
        return ApiSubscriptionResponse.fromEntity(subscription);
    }

    /**
     * Get all subscriptions for a developer
     */
    public List<ApiSubscriptionResponse> getDeveloperSubscriptions(String developerId) {
        return subscriptionRepository.findByDeveloperId(developerId).stream()
            .map(ApiSubscriptionResponse::fromEntity)
            .collect(Collectors.toList());
    }

    /**
     * Get all pending subscriptions
     */
    public List<ApiSubscriptionResponse> getPendingSubscriptions() {
        return subscriptionRepository.findByStatus(SubscriptionStatus.PENDING).stream()
            .map(ApiSubscriptionResponse::fromEntity)
            .collect(Collectors.toList());
    }

    /**
     * Get all active subscriptions
     */
    public List<ApiSubscriptionResponse> getActiveSubscriptions() {
        return subscriptionRepository.findByStatus(SubscriptionStatus.ACTIVE).stream()
            .map(ApiSubscriptionResponse::fromEntity)
            .collect(Collectors.toList());
    }

    /**
     * Check if developer has active subscription for an API pattern
     */
    public boolean hasActiveSubscription(String developerId, String apiPattern) {
        List<ApiSubscription> subscriptions = subscriptionRepository.findByDeveloperIdAndStatus(
            developerId, SubscriptionStatus.ACTIVE
        );
        
        return subscriptions.stream()
            .anyMatch(sub -> matchPath(sub.getApiPattern(), apiPattern));
    }

    /**
     * Match API path against pattern
     */
    private boolean matchPath(String pattern, String path) {
        if (pattern.endsWith("/**")) {
            String prefix = pattern.substring(0, pattern.length() - 3);
            return path.startsWith(prefix);
        }
        return pattern.equals(path);
    }

    /**
     * Generate unique subscription ID
     */
    private String generateSubscriptionId() {
        return "sub_" + UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }

    /**
     * Generate unique API key ID
     */
    private String generateKeyId() {
        return "ak_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24);
    }

    /**
     * Generate API key secret
     */
    private String generateKeySecret() {
        return UUID.randomUUID().toString().replace("-", "") + 
               UUID.randomUUID().toString().replace("-", "").substring(0, 8);
    }
}
