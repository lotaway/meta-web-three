package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.entity.ApiKey;
import com.metawebthree.developerportal.entity.ApiKey.KeyStatus;
import com.metawebthree.developerportal.repository.ApiDeveloperRepository;
import com.metawebthree.developerportal.repository.ApiKeyRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ApiKeyService {

    private final ApiKeyRepository apiKeyRepository;
    private final ApiDeveloperRepository developerRepository;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    @Transactional
    public ApiKeyResponse createApiKey(String developerId, ApiKeyCreateRequest request) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        if (developer.getStatus() != ApiDeveloper.DeveloperStatus.APPROVED) {
            throw new IllegalStateException("Developer is not approved: " + developer.getStatus());
        }

        String keyId = generateKeyId();
        String keySecret = generateKeySecret();

        ApiKey apiKey = new ApiKey();
        apiKey.setKeyId(keyId);
        apiKey.setKeySecret(passwordEncoder.encode(keySecret));
        apiKey.setDeveloperId(developerId);
        apiKey.setName(request.getName());
        apiKey.setStatus(KeyStatus.ACTIVE);
        apiKey.setExpiresAt(request.getExpiresAt());
        apiKey.setScopes(request.getScopes());
        apiKey.setAllowedIps(request.getAllowedIps());
        apiKey.setAllowedDomains(request.getAllowedDomains());
        apiKey.setRateLimit(request.getRateLimit() != null ? request.getRateLimit() : 100);

        apiKeyRepository.save(apiKey);
        log.info("API Key created: {} for developer {}", keyId, developerId);

        ApiKeyResponse response = ApiKeyResponse.fromEntity(apiKey);
        response.setKeySecret(keySecret);
        return response;
    }

    public boolean validateApiKey(String keyId, String keySecret) {
        ApiKey apiKey = apiKeyRepository.findByKeyId(keyId).orElse(null);

        if (apiKey == null) {
            log.warn("API Key not found: {}", keyId);
            return false;
        }

        if (apiKey.getStatus() != KeyStatus.ACTIVE) {
            log.warn("API Key is not active: {} - {}", keyId, apiKey.getStatus());
            return false;
        }

        if (apiKey.getExpiresAt() != null && apiKey.getExpiresAt().isBefore(LocalDateTime.now())) {
            log.warn("API Key has expired: {}", keyId);
            return false;
        }

        if (!passwordEncoder.matches(keySecret, apiKey.getKeySecret())) {
            log.warn("API Key secret mismatch: {}", keyId);
            return false;
        }

        apiKey.setLastUsedAt(LocalDateTime.now());
        apiKeyRepository.save(apiKey);

        return true;
    }

    public ApiKeyResponse getApiKey(String keyId) {
        ApiKey apiKey = apiKeyRepository.findByKeyId(keyId)
            .orElseThrow(() -> new IllegalArgumentException("API Key not found: " + keyId));
        return ApiKeyResponse.fromEntity(apiKey);
    }

    public List<ApiKeyResponse> getDeveloperApiKeys(String developerId) {
        return apiKeyRepository.findByDeveloperId(developerId).stream()
            .map(ApiKeyResponse::fromEntity)
            .collect(Collectors.toList());
    }

    @Transactional
    public ApiKeyResponse disableApiKey(String keyId) {
        ApiKey apiKey = apiKeyRepository.findByKeyId(keyId)
            .orElseThrow(() -> new IllegalArgumentException("API Key not found: " + keyId));

        apiKey.setStatus(KeyStatus.DISABLED);
        apiKeyRepository.save(apiKey);
        log.info("API Key disabled: {}", keyId);

        return ApiKeyResponse.fromEntity(apiKey);
    }

    @Transactional
    public ApiKeyResponse enableApiKey(String keyId) {
        ApiKey apiKey = apiKeyRepository.findByKeyId(keyId)
            .orElseThrow(() -> new IllegalArgumentException("API Key not found: " + keyId));

        apiKey.setStatus(KeyStatus.ACTIVE);
        apiKeyRepository.save(apiKey);
        log.info("API Key enabled: {}", keyId);

        return ApiKeyResponse.fromEntity(apiKey);
    }

    @Transactional
    public ApiKeyResponse revokeApiKey(String keyId) {
        ApiKey apiKey = apiKeyRepository.findByKeyId(keyId)
            .orElseThrow(() -> new IllegalArgumentException("API Key not found: " + keyId));

        apiKey.setStatus(KeyStatus.REVOKED);
        apiKeyRepository.save(apiKey);
        log.info("API Key revoked: {}", keyId);

        return ApiKeyResponse.fromEntity(apiKey);
    }

    @Transactional
    public ApiKeyResponse regenerateKeySecret(String keyId) {
        ApiKey apiKey = apiKeyRepository.findByKeyId(keyId)
            .orElseThrow(() -> new IllegalArgumentException("API Key not found: " + keyId));

        String newSecret = generateKeySecret();
        apiKey.setKeySecret(passwordEncoder.encode(newSecret));
        apiKeyRepository.save(apiKey);
        log.info("API Key secret regenerated: {}", keyId);

        ApiKeyResponse response = ApiKeyResponse.fromEntity(apiKey);
        response.setKeySecret(newSecret);
        return response;
    }

    public boolean isIpAllowed(ApiKey apiKey, String clientIp) {
        if (apiKey.getAllowedIps() == null || apiKey.getAllowedIps().isEmpty()) {
            return true;
        }

        String[] allowedIps = apiKey.getAllowedIps().split(",");
        for (String allowedIp : allowedIps) {
            if (allowedIp.trim().equals(clientIp)) {
                return true;
            }
        }

        return false;
    }

    private String generateKeyId() {
        return "ak_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24);
    }

    private String generateKeySecret() {
        return UUID.randomUUID().toString().replace("-", "") + 
               UUID.randomUUID().toString().replace("-", "").substring(0, 8);
    }
}
