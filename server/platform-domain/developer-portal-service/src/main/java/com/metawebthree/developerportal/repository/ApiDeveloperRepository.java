package com.metawebthree.developerportal.repository;

import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.entity.ApiDeveloper.DeveloperStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * API Developer Repository
 */
@Repository
public interface ApiDeveloperRepository extends JpaRepository<ApiDeveloper, Long> {

    /**
     * Find developer by developer ID
     */
    Optional<ApiDeveloper> findByDeveloperId(String developerId);

    /**
     * Find developer by email
     */
    Optional<ApiDeveloper> findByEmail(String email);

    /**
     * Find all developers by status
     */
    List<ApiDeveloper> findByStatus(DeveloperStatus status);

    /**
     * Check if email exists
     */
    boolean existsByEmail(String email);

    /**
     * Check if developer ID exists
     */
    boolean existsByDeveloperId(String developerId);

    /**
     * Find developers with balance below threshold
     */
    @org.springframework.data.jpa.repository.Query("SELECT d FROM ApiDeveloper d WHERE d.balance < :thresholdCents AND d.status = 'APPROVED'")
    List<ApiDeveloper> findByBalanceBelowThreshold(@org.springframework.data.repository.query.Param("thresholdCents") long thresholdCents);
}
