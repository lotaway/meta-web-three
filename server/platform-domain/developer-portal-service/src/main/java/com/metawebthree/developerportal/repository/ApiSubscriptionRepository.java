package com.metawebthree.developerportal.repository;

import com.metawebthree.developerportal.entity.ApiSubscription;
import com.metawebthree.developerportal.entity.ApiSubscription.SubscriptionStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * API Subscription Repository
 */
@Repository
public interface ApiSubscriptionRepository extends JpaRepository<ApiSubscription, Long> {

    /**
     * Find subscription by subscription ID
     */
    Optional<ApiSubscription> findBySubscriptionId(String subscriptionId);

    /**
     * Find all subscriptions by developer ID
     */
    List<ApiSubscription> findByDeveloperId(String developerId);

    /**
     * Find all subscriptions by developer ID and status
     */
    List<ApiSubscription> findByDeveloperIdAndStatus(String developerId, SubscriptionStatus status);

    /**
     * Find all subscriptions by status
     */
    List<ApiSubscription> findByStatus(SubscriptionStatus status);

    /**
     * Check if developer has subscription for API pattern
     */
    boolean existsByDeveloperIdAndApiPattern(String developerId, String apiPattern);
}
