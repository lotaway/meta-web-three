package com.metawebthree.developerportal.repository;

import com.metawebthree.developerportal.entity.ApiKey;
import com.metawebthree.developerportal.entity.ApiKey.KeyStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * API Key Repository
 */
@Repository
public interface ApiKeyRepository extends JpaRepository<ApiKey, Long> {

    /**
     * Find API key by key ID
     */
    Optional<ApiKey> findByKeyId(String keyId);

    /**
     * Find all API keys by developer ID
     */
    List<ApiKey> findByDeveloperId(String developerId);

    /**
     * Find all API keys by developer ID and status
     */
    List<ApiKey> findByDeveloperIdAndStatus(String developerId, KeyStatus status);

    /**
     * Check if key ID exists
     */
    boolean existsByKeyId(String keyId);

    /**
     * Find all active keys
     */
    List<ApiKey> findByStatus(KeyStatus status);
}
