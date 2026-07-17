package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.entity.ApiKey;
import com.metawebthree.developerportal.repository.ApiKeyRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Slf4j
@RestController
@RequestMapping("/developer/api-keys")
@RequiredArgsConstructor
public class ApiKeyValidationController {

    private final ApiKeyRepository apiKeyRepository;

    @GetMapping("/{keyId}/validate")
    public ResponseEntity<Map<String, Object>> validateApiKey(
            @PathVariable String keyId,
            @RequestHeader(value = "X-API-Secret", required = false) String apiSecret) {

        Map<String, Object> response = new HashMap<>();

        Optional<ApiKey> apiKeyOpt = apiKeyRepository.findByKeyId(keyId);

        if (apiKeyOpt.isEmpty()) {
            response.put("valid", false);
            response.put("message", "API key not found");
            return ResponseEntity.ok(response);
        }

        ApiKey apiKey = apiKeyOpt.get();

        if (apiKey.getStatus() != ApiKey.KeyStatus.ACTIVE) {
            response.put("valid", false);
            response.put("message", "API key is " + apiKey.getStatus().name().toLowerCase());
            return ResponseEntity.ok(response);
        }

        if (apiKey.getExpiresAt() != null && apiKey.getExpiresAt().isBefore(LocalDateTime.now())) {
            response.put("valid", false);
            response.put("message", "API key has expired");
            return ResponseEntity.ok(response);
        }

        if (apiSecret == null || apiSecret.isEmpty()) {
            response.put("valid", false);
            response.put("message", "API secret is required");
            return ResponseEntity.ok(response);
        }

        response.put("valid", true);
        response.put("developerId", apiKey.getDeveloperId());
        response.put("keyId", apiKey.getKeyId());
        response.put("scopes", apiKey.getScopes());
        response.put("rateLimit", apiKey.getRateLimit());

        apiKey.setLastUsedAt(LocalDateTime.now());
        apiKeyRepository.save(apiKey);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/{keyId}")
    public ResponseEntity<Map<String, Object>> getApiKey(@PathVariable String keyId) {
        Map<String, Object> response = new HashMap<>();

        Optional<ApiKey> apiKeyOpt = apiKeyRepository.findByKeyId(keyId);

        if (apiKeyOpt.isEmpty()) {
            response.put("error", "API key not found");
            return ResponseEntity.notFound().build();
        }

        ApiKey apiKey = apiKeyOpt.get();

        response.put("keyId", apiKey.getKeyId());
        response.put("developerId", apiKey.getDeveloperId());
        response.put("name", apiKey.getName());
        response.put("status", apiKey.getStatus().name());
        response.put("scopes", apiKey.getScopes());
        response.put("rateLimit", apiKey.getRateLimit());

        return ResponseEntity.ok(response);
    }
}
