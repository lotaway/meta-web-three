package com.metawebthree.developerportal.repository;

import com.metawebthree.developerportal.entity.OAuthApplication;
import com.metawebthree.developerportal.entity.OAuthApplication.AppStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * OAuth Application Repository
 */
@Repository
public interface OAuthApplicationRepository extends JpaRepository<OAuthApplication, Long> {

    /**
     * Find OAuth application by client ID
     */
    Optional<OAuthApplication> findByClientId(String clientId);

    /**
     * Find all OAuth applications by developer ID
     */
    List<OAuthApplication> findByDeveloperId(String developerId);

    /**
     * Find all OAuth applications by developer ID and status
     */
    List<OAuthApplication> findByDeveloperIdAndStatus(String developerId, AppStatus status);

    /**
     * Check if client ID exists
     */
    boolean existsByClientId(String clientId);
}
