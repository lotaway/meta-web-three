package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.service.ApiKeyService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * API Key Controller
 * Handles API key management for developers
 */
@Tag(name = "API Keys", description = "API key management for developers")
@RestController
@RequestMapping("/developer/api-keys")
@RequiredArgsConstructor
public class ApiKeyController {

    private final ApiKeyService apiKeyService;

    @Operation(summary = "Create API key", description = "Generate a new API key for authenticated developer")
    @PostMapping("/{developerId}")
    public ResponseEntity<ApiKeyResponse> createApiKey(
        @PathVariable String developerId,
        @Valid @RequestBody ApiKeyCreateRequest request
    ) {
        ApiKeyResponse response = apiKeyService.createApiKey(developerId, request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @Operation(summary = "Get API key", description = "Get API key details by key ID")
    @GetMapping("/{keyId}")
    public ResponseEntity<ApiKeyResponse> getApiKey(@PathVariable String keyId) {
        ApiKeyResponse response = apiKeyService.getApiKey(keyId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "List developer API keys", description = "Get all API keys for a developer")
    @GetMapping("/developer/{developerId}")
    public ResponseEntity<List<ApiKeyResponse>> getDeveloperApiKeys(@PathVariable String developerId) {
        List<ApiKeyResponse> response = apiKeyService.getDeveloperApiKeys(developerId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Disable API key", description = "Temporarily disable an API key")
    @PostMapping("/{keyId}/disable")
    public ResponseEntity<ApiKeyResponse> disableApiKey(@PathVariable String keyId) {
        ApiKeyResponse response = apiKeyService.disableApiKey(keyId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Enable API key", description = "Re-enable a disabled API key")
    @PostMapping("/{keyId}/enable")
    public ResponseEntity<ApiKeyResponse> enableApiKey(@PathVariable String keyId) {
        ApiKeyResponse response = apiKeyService.enableApiKey(keyId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Revoke API key", description = "Permanently revoke an API key")
    @PostMapping("/{keyId}/revoke")
    public ResponseEntity<ApiKeyResponse> revokeApiKey(@PathVariable String keyId) {
        ApiKeyResponse response = apiKeyService.revokeApiKey(keyId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Regenerate API key secret", description = "Generate a new secret for an API key")
    @PostMapping("/{keyId}/regenerate-secret")
    public ResponseEntity<ApiKeyResponse> regenerateKeySecret(@PathVariable String keyId) {
        ApiKeyResponse response = apiKeyService.regenerateKeySecret(keyId);
        return ResponseEntity.ok(response);
    }
}
